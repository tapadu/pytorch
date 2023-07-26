#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Half.h>
#include <cusparse.h>
#include <cstdint>
#include <Exceptions.h>


#if AT_CUSPARSELT_ENABLED()
#include <cusparseLt.h>
#endif

namespace at {
namespace native {

#if AT_CUSPARSELT_ENABLED()
cusparseLtHandle_t handle;
#endif

at::Tensor _cslt_compress(const Tensor& sparse_input)
{
#if AT_CUSPARSELT_ENABLED()

    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
    // create sparse descriptor, dtype
    cusparseLtMatDescriptor_t sparse_input_descriptor;
    cudaDataType type;
    auto compression_factor = 9;

    switch(
        sparse_input.scalar_type()
    )
    {
        case at::ScalarType::Char:
            type = CUDA_R_8I;
            compression_factor = 10;
            break;
        case at::ScalarType::Half:
            type = CUDA_R_16F;
            break;
        case at::ScalarType::BFloat16:
            type = CUDA_R_16BF;
            break;
        case at::ScalarType::Float:
            type = CUDA_R_32F;
            break;
        default:
            break;
    }

    // create a new compressed tensor with the same dtype as
    auto compressed_tensor = sparse_input.new_empty(sparse_input.numel() * compression_factor / 16);

    TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
        &handle,
        &sparse_input_descriptor,
        sparse_input.size(0),
        sparse_input.size(1),
        sparse_input.size(1),
        16,
        type,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    // compress input
    //--------------------------------------------------------------------------
    size_t compressed_size, compressed_buffer_size;
    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
        &handle,
        &sparse_input_descriptor,
        &compressed_size,
        &compressed_buffer_size));

    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);

    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
        &handle,
        &sparse_input_descriptor,
        true,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        sparse_input.data_ptr(),
        compressed_tensor.data_ptr(),
        compressedBufferPtr.get(),
        nullptr));

    return compressed_tensor;
#else
    TORCH_CHECK(false, "PyTorch must be compiled with cuSPARSELt to use _cslt_compress");
#endif
}


at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const c10::optional<Tensor>& bias_opt,
    bool transpose_result
)
{
#if AT_CUSPARSELT_ENABLED()
  // cupsarselt constructs
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  float alpha = 1.0;
  float beta = 0.0;
  cudaDataType type;
  cusparseComputeType compute_type;
  auto compression_factor = 9;

  switch(compressed_A.scalar_type())
  {
    case at::ScalarType::Char:
        type = CUDA_R_8I;
        compute_type = CUSPARSE_COMPUTE_32I;
        compression_factor = 10;
        break;
    case at::ScalarType::Half:
        type = CUDA_R_16F;
        compute_type = CUSPARSE_COMPUTE_16F;
        break;
    case at::ScalarType::BFloat16:
        type = CUDA_R_16BF;
        compute_type = CUSPARSE_COMPUTE_16F;
        break;
    case at::ScalarType::Float:
        type = CUDA_R_32F;
        compute_type = CUSPARSE_COMPUTE_TF32;
        break;
    default:
        TORCH_CHECK(false, "Unsupported dtype for cuSPARSE compressed matrix multiplication.")
        break;
  }

  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = (compressed_A.numel() * 16 / compression_factor  ) / k;

  //initialize sparse descriptor
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      k,
      16,
      type,
      CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  // initalize dense input descriptor
  cusparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      (dense_B.is_contiguous()) ? k : n,
      (dense_B.is_contiguous()) ? n : k,
      k,
      16,
      type,
      CUSPARSE_ORDER_ROW));

  // create result tensor
  auto res = (transpose_result) ? dense_B.new_empty({n, m})
                                : dense_B.new_empty({m, n});


  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      (transpose_result) ? m: n,
      16,
      type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // intialize matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &res_descriptor,
      &res_descriptor,
      compute_type));

  // set bias pointer for matmut, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)));
  }

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
      &handle,
      &plan,
      &alpha,
      compressed_A.data_ptr(),
      dense_B.data_ptr(),
      &beta,
      res.data_ptr(),
      res.data_ptr(),
      workspacePtr.get(),
      nullptr,
      0));


  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

  return res;
#else
        TORCH_CHECK(false, "PyTorch ust be compiled with cuSPARSELt to use _cslt_sparse_mm");
#endif
}

} // namespace native
} // namespace at
