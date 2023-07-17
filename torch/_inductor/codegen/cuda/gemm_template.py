from third_party.cutlass.tools.library import scripts as cutlass_lib

from .cuda_template import CutlassTemplate
from ..codegen.common import IndentedBuffer
from ...ir import IRNode, Layout

# Only supports alpha * A@B + beta * C now.
# TODO: Support arbitrary epilogue after epilogue visitor is released in cutlass 3.2.
GEMM_TEMPLATE = r"""
{{template.header().get_value()}}

{{template.globals().get_value()}}

{{template.define_gemm_instance(op, Bias)}}

template<typename GemmInstance>
{{define_kernel(X, W, Y, Bias, names="X, W, Y, Bias, GemmInstance gemm_op, uint8_t* workspace, cudaStream_t stream")}} {
  using ElementComputeEpilogue = GemmInstance::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  GemmInstance::Arguments arguments;

  {{kernel.check_not_null(X)}}
  {{kernel.check_not_null(W)}}
  {{kernel.check_not_null(Bias)}}
  {{kernel.check_not_null(Y)}}

  int64_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
  int64_t M = {{kernel.size(X, -2)}};
  int64_t K = {{kernel.size(X, -1)}};
  int64_t N = {{kernel.size(W, -1)}};

  {gemm_arguments}
}
"""


GEMM_ARGS_CUTLASS_2X = r"""
  int64_t batch_stride_x = {{kernel.stride(X, -3)}};
  int64_t row_stride_x = {{kernel.stride(X, -2, -1)}};
  int64_t offset_x = {{kernel.offset(X)}};

  int64_t batch_stride_w = {{kernel.stride(W, -3)}};
  int64_t row_stride_w = {{kernel.stride(W, -2, -1)}};
  int64_t offset_w = {{kernel.offset(W)}};

  int64_t batch_stride_bias = {{kernel.stride(Bias, -3)}};
  int64_t row_stride_bias = {{kernel.stride(Bias, -2, -1)}};
  int64_t offset_bias = {{kernel.offset(Bias)}};

  int64_t batch_stride_y = {{kernel.stride(Y, -3)}};
  int64_t row_stride_y = {{kernel.stride(Y, -2, -1)}};
  int64_t offset_y = {{kernel.offset(Y)}};

  // Initialize GemmUniversalInstance arguments.
  arguments = {
    {{template.gemm_mode(Y)}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K)
    },  // GemmCoord problem_size
    {{split_k if split_k > 1 else 'B'}},  // int batch_count
    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue
    ({{kernel.dtype(X)}}*)(X) + offset_x,  // void const * ptr_A
    ({{kernel.dtype(W)}}*)(W) + offset_w,  // void const * ptr_B
    ({{kernel.dtype(Bias)}}*)(Bias) + offset_bias,  // void const * ptr_C
    ({{kernel.dtype(Y)}}*)(Y) + offset_y,  // void * ptr_D
    batch_stride_x,  // int64_t batch_stride_A
    batch_stride_w,  // int64_t batch_stride_B
    batch_stride_bias,  // int64_t batch_stride_C
    batch_stride_y,  // int64_t batch_stride_D
    row_stride_x,  // typename LayoutA::Stride::LongIndex lda
    row_stride_w,  // typename LayoutB::Stride::LongIndex ldb
    row_stride_bias,  // typename LayoutC::Stride::LongIndex ldc
    row_stride_y,  // typename LayoutC::Stride::LongIndex ldd
  };
"""


GEMM_ARGS_CUTLASS_3X = r"""
  int64_t batch_stride_x = {{kernel.stride(X, -3)}};
  int64_t stride_x0 = {{kernel.stride(X, -2)}};
  int64_t stride_x1 = {{kernel.stride(X, -1)}};
  int64_t offset_x = {{kernel.offset(X)}};

  int64_t batch_stride_w = {{kernel.stride(W, -3)}};
  int64_t stride_w0 = {{kernel.stride(W, -2)}};
  int64_t stride_w1 = {{kernel.stride(W, -1)}};
  int64_t offset_w = {{kernel.offset(W)}};

  int64_t batch_stride_bias = {{kernel.stride(Bias, -3)}};
  int64_t stride_bias0 = {{kernel.stride(Bias, -2)}};
  int64_t stride_bias1 = {{kernel.stride(Bias, -1)}};
  int64_t offset_bias = {{kernel.offset(Bias)}};

  int64_t batch_stride_y = {{kernel.stride(Y, -3)}};
  int64_t stride_y0 = {{kernel.stride(Y, -2)}};
  int64_t stride_y1 = {{kernel.stride(Y, -1)}};
  int64_t offset_y = {{kernel.offset(Y)}};

  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    {{template.gemm_mode(Y)}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      ({{kernel.dtype(X)}}*)(X) + offset_x,  // ElementA const* ptr_A
      {stride_x0, stride_x1, batch_stride_x},  // StrideA dA
      ({{kernel.dtype(W)}}*)(W) + offset_w,  // ElementB const* ptr_B
      {stride_w0, stride_w1, batch_stride_w},  // StrideB dB
    },  // MainloopArguments mainloop
    {epilogue_arguments}
  };
"""


GEMM_ARGS_CUTLASS_3X_EPILOGUE_NO_TMA = r"""
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      ({{kernel.dtype(Bias)}}*)(Bias) + offset_bias,  // ElementC const* ptr_C
      {stride_bias0, stride_bias1, batch_stride_bias},  // StrideC dC
      ({{kernel.dtype(Y)}}*)(Y) + offset_y,  // ElementD const* ptr_D
      {stride_y1, stride_y0, batch_stride_y},  // StrideD dD
    },  // EpilogueArguments epilogue
"""


GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS_VECTOR = r"""
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      nullptr,  // ElementC const* ptr_C
      {cute::Int<1>{}, cute::Int<0>{}, cute::Int<0>{}},  // StrideC dC
      ({{kernel.dtype(Y)}}*)(Y) + offset_y,  // ElementD const* ptr_D
      {stride_y1, stride_y0, batch_stride_y},  // StrideD dD
      ({{kernel.dtype(Bias)}}*)(Bias) + offset_bias,  // ElementBias const* ptr_Bias
    },  // EpilogueArguments epilogue
"""


GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS_MATRIX = r"""
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      ({{kernel.dtype(Bias)}}*)(Bias) + offset_bias,  // ElementC const* ptr_C
      {stride_bias0, stride_bias1, batch_stride_bias},  // StrideC dC
      ({{kernel.dtype(Y)}}*)(Y) + offset_y,  // ElementD const* ptr_D
      {stride_y1, stride_y0, batch_stride_y},  // StrideD dD
      nullptr,  // ElementBias const* ptr_Bias
    },  // EpilogueArguments epilogue
"""


class CutlassGemmTemplate(CutlassTemplate):
    # Calculates alpha * X@W + beta * Bias
    def __init__(self, input_nodes: List[IRNode], layout: Layout, alpha: float, beta: float):
        super().__init__("GEMM", input_nodes, layout)
        self.alpha = alpha
        self.beta = beta


    def header(self) -> IndentedBuffer:
        return super().header().splice(
            """
                #include "cutlass/gemm/gemm.h"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/gemm/device/gemm_universal.h"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"
                #include "cutlass/gemm/kernel/gemm_universal.hpp"
            """
        )

    @staticmethod
    def cutlass_layout(torch_layout) -> Optional[cutlass_lib.library.LayoutType]:
        if torch_layout.stride[-1] == 1:
            return cutlass_lib.library.LayoutType.RowMajor
        elif torch_layout.stride[-2] == 1:
            return cutlass_lib.library.LayoutType.ColumnMajor
        else:
            return None


    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: cutlass_lib.library.LayoutType,
    ) -> cutlass_lib.library.LayoutType:
        if cutlass_layout == cutlass_lib.library.LayoutType.RowMajor:
            return cutlass_lib.library.LayoutType.ColumnMajor
        else:
            return cutlass_lib.library.LayoutType.RowMajor


    @staticmethod
    def layout_match(torch_layout, cutlass_layout) -> bool:
        return CutlassGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout


    @staticmethod
    def has_tma_epilogue(op) -> bool:
        result = False
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result


    @staticmethod
    def define_gemm_instance(
        op: cutlass_lib.gemm_operation.GemmOperation,
    ) -> str:
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            emitter = cutlass_lib.gemm_operation.EmitGemmUniversal3xInstance()
            op_def = emitter.emit(op)
        else:
            emitter = cutlass_lib.gemm_operation.EmitGemmInstance()
            op_def = emitter.emit(op)
            op_def = op_def.replace(
                "cutlass::gemm::device::Gemm", "cutlass::gemm::device::GemmUniversal"
            )
            op_def= op_def.replace("false,", "")
        return op_def


    @staticmethod
    def should_treat_bias_as_tma_matrix(
        op: cutlass_lib.gemm_operation.GemmOperation,
        bias: IRNode,
        beta: float,
    ) -> bool:
        has_tma_epilogue = CutlassGemmTemplate.has_tma_epilogue(op)
        has_bias = bias is not None
        is_bias_matrix = (len(bias.get_size()) > 1)

        return has_tma_epilogue and has_bias and (beta != 1.0 or is_bias_matrix)


    @staticmethod
    def swap_XW(
        op: cutlass_lib.gemm_operation.GemmOperation,
        X: IRNode,
        W: IRNode,
        Y: IRNode,
        Bias: IRNode,
    ) -> (cutlass_lib.gemm_operation.GemmOperation, IRNode, IRNode, IRNode, IRNode):
        # Swap X and W in GemmOperation.
        new_op = copy.deepcopy(op)
        new_op.A.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.A.layout)
        new_op.B.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.B.layout)
        new_op.A, new_op.B = new_op.B, new_op.A
        new_op.C.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.C.layout)
        new_op.D.layout = CutlassGemmTemplate.flip_cutlass_layout(new_op.D.layout)

        # Swap
        new_Bias = copy.deepcopy(Bias)
        Bias_layout = new_Bias.get_layout()
        if len(Bias_layout.stride) == 1:
            Bias_layout.stride = [1] + Bias_layout.stride
        Bias_layout.stride[-2], Bias_layout.stride[-1] = Bias_layout.stride[-1], Bias_layout.stride[-2]
        new_Y = copy.deepcopy(Y)
        Y_layout = new_Y.get_layout()
        Y_layout.stride[-2], Y_layout.stride[-1] = Y_layout.stride[-1], Y_layout.stride[-2]

        return (new_op, W, X, new_Y, new_bias)


    def filter_op(
        self,
        op: cutlass_layout.gemm_operation.GemmOperation,
    ) -> cutlass_lib.gemm_operation.GemmOperation:
        ret = []

        # skip simt kernels
        if (
            op.tile_description.math_instruction.opcode_class
            == cutlass_lib.library.OpcodeClass.Simt
        ):
            return ret

        # Filter ops by dtypes.
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        if not (
            cutlass_utils.dtype_match(X.get_dtype(), op.A.element) and
            cutlass_utils.dtype_match(W.get_dtype(), op.B.element) and
            cutlass_utils.dtype_match(self.output_node.layout.dtype, op.C.element)
        ):
            return ret

        # Filter ops by accumulation type.
        if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
            if not cutlass_utils.dtype_match(torch.float16, op.accumulator_type):
                return ret

        # Filter ops by input layouts.
        if not (
            self.layout_match(X.get_layout(), op.A.layout) and
            self.layout_match(W.get_layout(), op.B.layout)
        ):
            return ret

        # Update op.
        op = copy.deepcopy(op)

        # Set output layout and alignment.
        def set_output_alignment(torch_layout, op_element, has_tma_epilogue, name) -> bool:
            cutlass_layout = self.cutlass_layout(torch_layout)
            if cutlass_layout is None:
                raise RuntimeError(f"Unsupported {name} layout in cutlass: {torch_layout}!")
            op_element.layout = outout_layput
            op_element.alignment = cutlass_utils.get_alignment(torch_layout)
            if has_tma_epilogue:
                max_alignment = cutlass_utils.get_alignment(self.output_node.layout.dtype)
                if op_element.alignment != max_alignment:
                    return False
            return True

        has_tma_epilogue = self.has_tma_epilogue(op)
        if set_output_alignment(self.output_node.layout, op.D, has_tma_epilogue, "output"):
            return ret

        # Set epilogue.
        # TODO: update epilogue functor according to epilogues.
        op.element_epilogue = op.accumulator_type

        # Set bias layout and alignment.
        if len(self.input_nodes) == 3:
            Bias = self.input_nodes[2]
            if set_output_alignment(Bias.get_layout(), op.C, has_tma_epilogue, "bias"):
                return ret

        return ret


    def gen_ops(self) -> List[cutlass_lib.gemm_operation.GemmOperation]:
        ops = cutlass_utils.gen_ops()[cutlass_lib.library.OperationKind.Gemm].values()
        ops = [op for op in ops if self.filter_op(op, self.input_nodes, self.output_node.layout)]
        return ops


    def gemm_mode(self) -> str:
        sizes = self.output_node.get_size()
        if len(sizes) > 2:
            return "cutlass::gemm::GemmUniversalMode::kBatched"
        else:
            return "cutlass::gemm::GemmUniversalMode::kGemm"


    def render(
        self,
        kernel: CUDATemplateKernel,
        op: cutlass_lib.gemm_operation.GemmOperation,
    ) -> str:
        assert len(self.input_nodes) >= 2 and self.output_node is not None
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        options = dict(
            alpha=self.alpha,
            beta=self.beta,
            X=X,
            W=W,
            Y=Y,
            Bias=Bias,
            template=self,
            kernel=kernel,
        )
        arguments = ""
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            epilogue_arguments = ""
            if self.has_tma_epilogue(op):
                if self.should_treat_bias_as_tma_matrix(op, bias, beta):
                    epilogue_arguments = self._template_from_string(GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS_MATRIX).render(**options)
                else:
                    # TMA epilogue requires bias vector in column major to get best perf.
                    if op.D.layout == cutlass_lib.library.LayoutType.RowMajor:
                        options[op], options[X], options[W], options[Y], options[Bias] = self.maybe_swap_XW(op, X, W, Y, Bias)
                    epilogue_arguments = self._template_from_string(GEMM_ARGS_CUTLASS_3X_EPILOGUE_TMA_BIAS).render(**options)
            else:
                epilogue_arguments = self._template_from_string(GEMM_ARGS_CUTLASS_3X_EPILOGUE_NO_TMA).render(**options)
            arguments = self._template_from_string(GEMM_ARGS_CUTLASS_3X).render(epilogue_arguments=epilogue_arguments, **options)
        else:
            # TODO: Support split_k.
            arguments = self._template_from_string(GEMM_ARGS_CUTLASS_2X).render(split_k=1, **options)
        res = self._template_from_string(GEMM_TEMPLATE).render(arguments=arguments, **options)
        return res
