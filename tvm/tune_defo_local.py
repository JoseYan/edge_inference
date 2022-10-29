# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-scheduling a Neural Network for x86 CPU
============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole neural
network for x86 CPU with the auto-scheduler.

To auto-tune a neural network, we partition the network into small subgraphs and
tune them independently. Each subgraph is treated as one search task.
A task scheduler slices the time and dynamically allocates time resources to
these tasks. The task scheduler predicts the impact of each task on the end-to-end
execution time and prioritizes the one that can reduce the execution time the most.

For each subgraph, we use the compute declaration in :code:`tvm/python/topi` to
get the computational DAG in the tensor expression form.
We then use the auto-scheduler to construct a search space of this DAG and search
for good schedules (low-level optimizations).

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any
schedule templates. In other words, the auto-scheduler only uses the compute declarations
in :code:`tvm/python/topi` and does not use existing schedule templates.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

import numpy as np

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
import time
from timm.models import  mobilevitv2_075, beit_base_patch16_224,mobilevit_xxs,edgenext_xx_small, vit_tiny_patch16_224, vit_base_patch16_224
from timm.models.vision_transformer import VisionTransformer


from torchvision import transforms
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# For convolutional neural networks, although auto-scheduler can work correctly
# with any layout, we found the best performance is typically achieved with NHWC layout.
# We also implemented more optimizations for NHWC layout with the auto-scheduler.
# So it is recommended to convert your models to NHWC layout to use the auto-scheduler.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.


def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        #image_shape = (256, 256, 3)
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        #image_shape = (3, 256, 256)
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 2)
    #output_shape = (batch_size, 1000)

    trans = None
    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    elif "defo" in name  :
        from defo_model import DefoNet
        if 'dw' in name:
            use_dw = True
        else:
            use_dw = False
        _, sf, _ = name.strip().split('_')
        sf = int(sf)
        model = DefoNet(3,2, scale_factor=sf, use_dw=use_dw)
        import torch
        input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        model.eval()
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])

        #DefoNet(3, 2, scale_factor=i, use_dw=False)
    elif name in ['vit_base_16_112', 'vit_base_8_112']:
        import torch
        if '16' in name:
            model = VisionTransformer(img_size=112,patch_size=16,num_classes=2)
        else:
            model = VisionTransformer(img_size=112,patch_size=8, num_classes=2)
        #print(model)
        #input_shape = [1, 3, 256, 256]
        input_shape = [1, 3, 112, 112]
        #input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        trans = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        #for _ in range(1000):
        #    scripted_model(input_data)
        #el = time.time() -st
        #print('TorchScript exec time:', f'{el:.4f}')
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name in ['vit_tiny_224', 'vit_base_224']:
        import torch
        if 'tiny' in name:
            model =vit_tiny_patch16_224(pretrained=False, num_classes=2)
        else:
            model =vit_base_patch16_224(pretrained=False, num_classes=2)
        #print(model)
        #input_shape = [1, 3, 256, 256]
        input_shape = [1, 3, 224, 224]
        #input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        #for _ in range(1000):
        #    scripted_model(input_data)
        #el = time.time() -st
        #print('TorchScript exec time:', f'{el:.4f}')
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name =='vit_tiny_16_112':
        import torch
        model = VisionTransformer(img_size=112,patch_size=16,embed_dim=192, depth=12, num_heads=3,num_classes=2)
        #print(model)
        #input_shape = [1, 3, 256, 256]
        input_shape = [1, 3, 112, 112]
        #input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        trans = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        #for _ in range(1000):
        #    scripted_model(input_data)
        #el = time.time() -st
        #print('TorchScript exec time:', f'{el:.4f}')

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name =='vit_tiny_8_112':
        import torch
        model = VisionTransformer(img_size=112,patch_size=8,embed_dim=192, depth=12, num_heads=3,num_classes=2)
        #print(model)
        #input_shape = [1, 3, 256, 256]
        input_shape = [1, 3, 112, 112]
        #input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        trans = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])

       #for _ in range(1000):
        #    scripted_model(input_data)
        #el = time.time() -st
        #print('TorchScript exec time:', f'{el:.4f}')

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name =='beit_base':
        import torch
        model = beit_base_patch16_224(pretrained=False, num_classes=2)
        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name =='mobilevit_108':
        import torch
        model = mobilevitv2_075(pretrained=False, num_classes=2)
        input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name =='mobilevit_256':
        import torch
        model = mobilevitv2_075(pretrained=False, num_classes=2)
        input_shape = [1, 3, 256, 256]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ])
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 'lin':
        import torch
        model = torch.nn.Linear(768, 768*4)
        input_shape = [16, 768]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 'faclin':
        import tltorch
        import torch
        #model = torch.nn.Linear(768, 768*4)
        model = tltorch.factorized_layers.FactorizedLinear((384,2),(768,4), bias=True, factorization='cp', rank='same')
        input_shape = [16, 768]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)

    ### Benchmarking pytorch model with image io

    #ims1 = [str(i) for i in list(Path(pref+'DefoNet/nn_Data_Set_Cropped/test/1').rglob('*.jpg'))]
    #st = time.time()
    #for im in tqdm(ims[0:1000]):
    #    im = Image.open(im)
    #    im = torch.unsqueeze(trans(im), dim=0)
    #    #im = torch.randn(16,768)
    #    model(im)
    #el = time.time() -st
    #print('PyTorch exec time:', f'{el:.4f}')
    #log = f'{name},PyTorch:{el:.4f}\n'
    #with open('TVM_TUNE_LOG.csv', 'a') as f:
    #    f.write(log)

    #from torch.utils import mkldnn as mkldnn_utils
    #### this will recursively convert `weight` and `bias` to MkldnnTensor and do weight prepacking
    #model_mkl = mkldnn_utils.to_mkldnn(model)
    #st = time.time()
    #for im in tqdm(ims[0:1000]):
    #    im = Image.open(im)
    #    im = torch.unsqueeze(trans(im), dim=0).to_mkldnn()
    #    #im = torch.randn(16,768)
    #    model_mkl(im)
    #el = time.time() -st
    #print('MKL exec time:', f'{el:.4f}')
    #log = f'{name},mkl:{el:.4f}\n'
    #with open('TVM_TUNE_LOG.csv', 'a') as f:
    #    f.write(log)
    return mod, params, input_shape, output_shape, trans

# Define the neural network and compilation target.
# If the target machine supports avx512 instructions, replace the
# "llvm -mcpu=core-avx2" with "llvm -mcpu=skylake-avx512"
import sys
network = sys.argv[1]
#network = "defo"
#network = "resnet-50"
use_sparse = False
batch_size = 1
layout = "NHWC"
arch = sys.argv[2]
if arch == 'neon':
    target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
elif arch== 'cortex':
    target = tvm.target.Target("llvm -mcpu=cortex-a72")
elif arch == 'znver2':
    target = tvm.target.Target("llvm -mcpu=znver2")
elif arch == 'avx2':
    target = tvm.target.Target("llvm -mcpu=core-avx2")
elif arch == 'avx512':
    target = tvm.target.Target("llvm -mcpu=skylake-avx512")
elif arch == 'llvm':
    target = tvm.target.Target("llvm")
else:
    raise NotImplementedError
print(f'Using llvm {arch}')

dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)

#################################################################
# Extract Search Tasks
# --------------------
# Next, we extract the search tasks and their weights from a network.
# The weight of a task is the number of appearances of the task's subgraph
# in the whole network.
# By using the weight, we can approximate the end-to-end latency of the network
# as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
# latency of a task and :code:`weight[t]` is the weight of the task.
# The task scheduler will just optimize this objective.

# Extract tasks from the network
print("Get model...")
mod, params, input_shape, output_shape,trans = get_network(
    network,
    batch_size,
    layout,
    dtype=dtype,
    use_sparse=use_sparse,
)
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
#exit()
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
#################################################################
# Begin Tuning
# ------------

def run_tuning():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
            for task in tasks
        ]

        tuner.tune(tune_option, search_policy=search_policy)
    else:
        tuner.tune(tune_option)


# Compile with the history best
print("Compile...")
#with open(log_file,'r') as f:
#    print(f.readline())
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("input0", data_tvm)

# Evaluate
#print("Evaluate inference time cost...")
#line = str(module.benchmark(dev, repeat=3, min_repeat_ms=500))
#print(line)
#with open('TVM_TUNE_LOG.csv', 'a') as f:
#    f.write('****Untuned*******\n'+line)
#
print('Run tuning')
run_tuning()

print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("input0", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
line = str(module.benchmark(dev, repeat=3, min_repeat_ms=500))
print(line)
#with open('TVM_TUNE_LOG.csv', 'a') as f:
#    f.write('*****Tuned:*****\n'+line)
