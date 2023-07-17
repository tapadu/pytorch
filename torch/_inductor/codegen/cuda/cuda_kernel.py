from typing import Optional

from ..common import Kernel
from ..cpp import DTYPE_TO_CPP
from ...ir import IRNode, TemplateBuffer, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index < 0 else index + total_length


class CUDAKernel(Kernel):
    pass


class CUDATemplateKernel(CUDAKernel):
    def __init__(
        self,
        kernel_name,
    ):
        super().__init__()
        self.kernel_name = kernel_name
        self.named_nodes = {}


    def check_not_null(self, node: IRNode) -> str:
        size_str = self.size(node, 0, -1)
        name_str = self.args.input_buffers.get(node.get_name())

        res = """
        {
            if (!{name_str}) {
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {
                    throw std::runtime_error("input {name_str} is null!");
                }
            }
        }
        """.format(name_str=name_str, size_str=size_str)
        return res


    def def_kernel(self, *nodes: IRNode, names_str: str = "") -> str:
        """
        Hook called from template code to generate function def and
        needed args.
        """

        names = [x.strip() for x in names_str.strip().split(",")]
        if len(nodes) > len(names):
            raise RuntimeError(f"{len(nodes)=} > {len(names)=}, {nodes=}, {names=}")

        for name, node in zip(names[len(nodes)], nodes):
            self.named_nodes[name] = node
            self.args.input_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs()
        arg_defs = arg_defs + names[len(nodes):]
        return f"void {self.kernel_name} ({', '.join(arg_defs)})"


    def dtype(self, node: IRNode) -> str:
        return DTYPE_TO_CPP.get(node.get_layout().dtype)


    def offset(self, node: IRNode) -> str:
        return str(node.get_layout().offset)


    def size(self, node: IRNode, start_index: int, end_index: Optional[int] = None, default_value: int = 0) -> str:
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = normalize_idx(end_index, len(node.get_size()))

        sizes = node.get_size()[start_index : end_index + 1]
        if len(sizes) == 0:
            return str(default_value)

        val = sympy_product(sizes)
        return texpr(self.rename_indexing(val))


    def stride(self, node: IRNode, start_index: int, end_index: Optional[int] = None, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        start_index = normalize_idx(start_index, len(node.get_stride()))
        if end_index is None:
            end_index = start_index
        end_index = normalize_idx(end_index, len(node.get_stride()))

        strides = node.get_stride()[start_index : end_index + 1]
        if len(strides) == 0:
            return str(default_value)

        val = Sympy.Max(strides)
        return texpr(self.rename_indexing(val))


class CUDATemplateCaller(ChoiceCaller):
    def __init__(
        self, name, input_nodes, layout, make_kernel_render, bmreq
    ):
        super().__init__(name, input_nodes, layout)
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        return f"CUDATemplateCaller({self.bmreq.module_path})"

    def call_name(self):
        return f"template_kernels.{self.name}"

    def hash_key(self):
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def output_node(self):
        return TensorBox.create(
            TemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
            )
        )
