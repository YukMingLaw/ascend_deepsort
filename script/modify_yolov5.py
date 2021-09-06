import onnx
import numpy as np
import sys
model_path = sys.argv[1]
mode = sys.argv[2]
onnx_model = onnx.load(model_path)
graph = onnx_model.graph

if mode == 'yolov5s':
    # yolov5s
    for i in range(8,-1,-1):
        node = graph.node[i]
        print(node.name)
        graph.node.remove(node)
elif mode == 'yolov5m'
    #yolov5m
    for i in range(40,-1,-1):
        node = graph.node[i]
        print(node.name)
        graph.node.remove(node)
else:
    exit()

weight=np.array(
    [
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]]],

        [[[0, 0], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [1, 0]]],

        [[[0, 1], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 1], [0, 0]]],

        [[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 1]]]
    ], dtype=np.float32)

SliceConvWeight = onnx.helper.make_tensor('SliceConvWeight', onnx.TensorProto.FLOAT, [12,3,2,2], weight.flatten())
graph.initializer.append(SliceConvWeight)
new_node = onnx.helper.make_node(
    'Conv',
    name='SliceConv',
    inputs=['images', 'SliceConvWeight'],
    outputs=['SliceOut'],
    dilations=[1,1],
    group=1,
    kernel_shape=[2,2],
    pads=[0,0,0,0],
    strides=[2,2]
)
graph.node.insert(0,new_node)

for node in graph.node:
    if node.name=='Conv_41':
        print(node)
        node.input[0]='SliceOut'


onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'yolov5_modified.onnx')
