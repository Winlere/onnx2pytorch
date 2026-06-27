import importlib.util

import onnx
import torch

from onnx2pytorch.convert.operations import convert_operations
from onnx2pytorch.operations.add import Add


def _single_node_graph(node):
    return onnx.helper.make_graph(
        [node],
        "single_node",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1]),
        ],
        [
            onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1]),
        ],
    )


def _converted_op(node, quirks=None):
    converted = list(convert_operations(
        _single_node_graph(node),
        opset_version=13,
        quirks=quirks or {},
    ))
    assert len(converted) == 1
    return converted[0][2]


def test_quantized_ops_are_disabled_by_default():
    node = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"])

    op = _converted_op(node)

    assert isinstance(op, Add)


def test_quantized_ops_are_enabled_by_explicit_quirk():
    from onnx2pytorch.operations.quantized_ops import BroadcastAdd

    node = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"])

    op = _converted_op(node, {"Quantized": {"enabled": True}})

    assert isinstance(op, BroadcastAdd)


def test_qlinearconv_policy_comes_from_explicit_quirks():
    from onnx2pytorch.operations.quantized_ops import QLinearConv

    node = onnx.helper.make_node(
        "QLinearConv",
        inputs=[
            "x", "x_scale", "x_zero", "w", "w_scale", "w_zero",
            "y_scale", "y_zero",
        ],
        outputs=["/encoder/conv1/output"],
    )

    op = _converted_op(
        node,
        {
            "Quantized": {
                "enabled": True,
                "qlinearconv_grad_mode": "auto",
                "qlinearconv_surrogate_patterns": ["QLinearConv_"],
                "qlinearconv_exact_patterns": ["/encoder/conv1/"],
            }
        },
    )

    assert isinstance(op, QLinearConv)
    assert op.grad_mode == "auto"
    assert op.force_exact_in_pgd is True


def test_qlinearconv_uses_default_safe_node_policy():
    from onnx2pytorch.operations.quantized_ops import QLinearConv

    risky_node = onnx.helper.make_node(
        "QLinearConv",
        inputs=[
            "x", "x_scale", "x_zero", "w", "w_scale", "w_zero",
            "y_scale", "y_zero",
        ],
        outputs=["/encoder/conv1/output"],
    )
    safe_node = onnx.helper.make_node(
        "QLinearConv",
        inputs=[
            "x", "x_scale", "x_zero", "w", "w_scale", "w_zero",
            "y_scale", "y_zero",
        ],
        outputs=["/layer4/layer4.0/conv1/output"],
    )

    risky_op = _converted_op(risky_node, {"Quantized": {"enabled": True}})
    safe_op = _converted_op(safe_node, {"Quantized": {"enabled": True}})

    assert isinstance(risky_op, QLinearConv)
    assert risky_op.grad_mode == "auto"
    assert risky_op.force_exact_in_pgd is True
    assert isinstance(safe_op, QLinearConv)
    assert safe_op.grad_mode == "auto"
    assert safe_op.force_exact_in_pgd is False


def test_qlinearmatmul_policy_comes_from_explicit_quirks():
    from onnx2pytorch.operations.quantized_ops import QLinearMatMul

    node = onnx.helper.make_node(
        "QLinearMatMul",
        inputs=[
            "a", "a_scale", "a_zero", "b", "b_scale", "b_zero",
            "y_scale", "y_zero",
        ],
        outputs=["/encoder/layers.0/self_attn/MatMul_output_0"],
    )

    op = _converted_op(
        node,
        {
            "Quantized": {
                "enabled": True,
                "qlinearmatmul_mode": "auto",
                "qlinearmatmul_torch_patterns": [
                    "self_attn/MatMul_output_0",
                ],
            }
        },
    )

    assert isinstance(op, QLinearMatMul)
    assert op.mode == "auto"
    assert op.force_torch_in_pgd is True


def test_qlinearmatmul_uses_default_safe_node_policy():
    from onnx2pytorch.operations.quantized_ops import QLinearMatMul

    inputs = [
        "a", "a_scale", "a_zero", "b", "b_scale", "b_zero",
        "y_scale", "y_zero",
    ]
    safe_outputs = [
        f"/encoder/layers.{layer}/self_attn/MatMul_output_0"
        for layer in range(4)
    ] + [
        f"/encoder/layers.{layer}/self_attn/MatMul_1_output_0"
        for layer in range(4)
    ]
    torch_ops = [
        _converted_op(
            onnx.helper.make_node(
                "QLinearMatMul", inputs=inputs, outputs=[output]),
            {"Quantized": {"enabled": True}},
        )
        for output in safe_outputs
    ]
    exact_node = onnx.helper.make_node(
        "QLinearMatMul", inputs=inputs,
        outputs=["/encoder/layers.0/linear1/MatMul_output_0"],
    )
    unsafe_self_attn_node = onnx.helper.make_node(
        "QLinearMatMul", inputs=inputs,
        outputs=["/encoder/layers.4/self_attn/MatMul_output_0"],
    )

    exact_op = _converted_op(exact_node, {"Quantized": {"enabled": True}})
    unsafe_self_attn_op = _converted_op(
        unsafe_self_attn_node, {"Quantized": {"enabled": True}})

    for torch_op in torch_ops:
        assert isinstance(torch_op, QLinearMatMul)
        assert torch_op.mode == "auto"
        assert torch_op.force_torch_in_pgd is True
    assert isinstance(exact_op, QLinearMatMul)
    assert exact_op.mode == "auto"
    assert exact_op.force_torch_in_pgd is False
    assert isinstance(unsafe_self_attn_op, QLinearMatMul)
    assert unsafe_self_attn_op.mode == "auto"
    assert unsafe_self_attn_op.force_torch_in_pgd is False


def test_qlinearmatmul_torch_mode_uses_torch_forward_with_gradients():
    from onnx2pytorch.operations.quantized_ops import QLinearMatMul

    op = QLinearMatMul(mode="torch")
    a = torch.tensor([[1.0, 2.0]], requires_grad=True)
    b = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    out = op(
        a,
        torch.tensor(0.5),
        torch.tensor(1, dtype=torch.uint8),
        b,
        torch.tensor(0.25),
        torch.tensor(2, dtype=torch.int8),
        torch.tensor(0.125),
        torch.tensor(0, dtype=torch.uint8),
    )

    assert torch.equal(out, torch.tensor([[3.0, 4.0]]))
    out.sum().backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert a.grad.abs().sum() > 0


def test_qlinearadd_always_uses_torch_without_mode_selection():
    from onnx2pytorch.operations.quantized_ops import QLinearAdd

    node = onnx.helper.make_node(
        "QLinearAdd",
        inputs=[
            "a", "a_scale", "a_zero", "b", "b_scale", "b_zero",
            "y_scale", "y_zero",
        ],
        outputs=["z"],
    )

    op = _converted_op(
        node,
        {
            "Quantized": {
                "enabled": True,
                "qlinearadd_exact_mode": "ort",
            }
        },
    )

    assert isinstance(op, QLinearAdd)
    assert not hasattr(op, "exact_mode")


def test_quantized_ops_do_not_include_runtime_profile_hooks():
    import onnx2pytorch.operations.quantized_ops as quantized_ops

    assert not hasattr(quantized_ops, "_profile_call")
    assert not hasattr(quantized_ops, "_PROFILE_STATS")


def test_quantized_uses_clear_module_names():
    assert importlib.util.find_spec(
        "onnx2pytorch.operations.quantized_ops") is not None
    assert importlib.util.find_spec(
        "onnx2pytorch.operations.quantized_ort_helpers") is not None
    assert importlib.util.find_spec(
        "onnx2pytorch.operations.quantized_ort_ops") is None
    assert importlib.util.find_spec(
        "onnx2pytorch.operations.quantized_ort_quantized") is None


def test_quantized_ops_file_only_exposes_wrappers_and_factory_helpers():
    import onnx2pytorch.operations.quantized_ops as quantized_ops

    forbidden = [
        "_ORTErfOp",
        "_ORTTanhOp",
        "_ORTSoftmaxOp",
        "_ORTReduceOp",
        "_ORTGlobalAveragePoolOp",
        "_ORTLayerNormOp",
        "_ort_erf_session",
        "_ort_tanh_session",
        "_ort_softmax_session",
        "_ort_reduce_session",
        "_ort_global_average_pool_session",
        "_ort_layer_norm_session",
        "_qlinear_add_exact_torch",
        "_qlinear_add_surrogate_torch",
        "_torch_quantize_ste",
        "_torch_dequantize",
    ]

    for name in forbidden:
        assert not hasattr(quantized_ops, name)
