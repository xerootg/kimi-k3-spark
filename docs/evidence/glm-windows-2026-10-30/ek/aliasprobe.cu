#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
static long avail_mb() { FILE*f=fopen("/proc/meminfo","r"); char l[256]; long v=-1; while(fgets(l,256,f)) if(!strncmp(l,"MemAvailable:",13)){sscanf(l+13,"%ld",&v);break;} fclose(f); return v/1024; }
#define CK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorString(r,&s);printf("%s -> %s\n",#x,s);return 1;}}while(0)
__global__ void rd(const unsigned long long *p, size_t n, unsigned long long *out){ unsigned long long s=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) s|=p[i]; if(s) atomicOr(out,s);}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){
  size_t GB = argc>1? atol(argv[1]):64;
  cudaFree(0); CUdevice d; CK(cuDeviceGet(&d,0));
  CUmemAllocationProp p={}; p.type=CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type=CU_MEM_LOCATION_TYPE_DEVICE; p.location.id=0;
  size_t g=0; CK(cuMemGetAllocationGranularity(&g,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  const size_t VA=GB<<30; CUdeviceptr base; CK(cuMemAddressReserve(&base,VA,0,0,0));
  long a0=avail_mb(); size_t n=VA/g; printf("granularity %zu, reserved %zu GiB VA (%zu chunks), avail %ld\n",g,GB,n,a0);
  CUmemGenericAllocationHandle dummy; CK(cuMemCreate(&dummy,g,&p,0));
  CUmemAccessDesc ad={}; ad.location=p.location; ad.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  double t0=now();
  for(size_t i=0;i<n;i++){ CK(cuMemMap(base+i*g,g,0,dummy,0)); }
  double t1=now();
  CK(cuMemSetAccess(base,VA,&ad,1));
  double t2=now();
  printf("alias map %.1f ms (%.2f us/chunk), setaccess whole range %.1f ms, avail %ld (delta %ld)\n",t1-t0,(t1-t0)*1000/n,t2-t1,avail_mb(),avail_mb()-a0);
  cudaMemset((void*)base,0,g); cudaDeviceSynchronize();
  unsigned long long *out; cudaMallocManaged(&out,8); *out=0;
  t0=now(); rd<<<1024,256>>>((unsigned long long*)base,VA/8,out); cudaError_t e=cudaDeviceSynchronize(); t1=now();
  printf("read whole VA: %s, or=%llx, %.1f ms (%.1f GB/s)\n",cudaGetErrorString(e),*out,t1-t0,VA/1e6/(t1-t0));
  // write through one alias, read through another
  cudaMemset((void*)(base+5*g+64),0xab,8); cudaDeviceSynchronize();
  unsigned long long v=0; cudaMemcpy(&v,(void*)(base+777*g+64),8,cudaMemcpyDeviceToHost); printf("alias write visible elsewhere: %llx\n",v);
  cudaMemset((void*)base,0,g); cudaDeviceSynchronize();
  // swap first 4096 chunks (8 GiB) to real
  size_t m=4096<n?4096:n/2;
  t0=now();
  for(size_t i=0;i<m;i++) CK(cuMemUnmap(base+i*g,g));
  t1=now();
  for(size_t i=0;i<m;i++){ CUmemGenericAllocationHandle h; CK(cuMemCreate(&h,g,&p,0)); CK(cuMemMap(base+i*g,g,0,h,0)); CK(cuMemRelease(h)); }
  CK(cuMemSetAccess(base,m*g,&ad,1)); cudaMemset((void*)base,0,m*g); cudaDeviceSynchronize();
  t2=now();
  printf("swap %zu chunks: unmap %.1f ms, map real+access+memset %.1f ms, avail %ld (delta %ld)\n",m,t1-t0,t2-t1,avail_mb(),avail_mb()-a0);
  *out=0; t0=now(); rd<<<1024,256>>>((unsigned long long*)base,VA/8,out); e=cudaDeviceSynchronize(); t1=now();
  printf("read whole VA after swap: %s, or=%llx, %.1f ms\n",cudaGetErrorString(e),*out,t1-t0);
  // swap back
  t0=now();
  for(size_t i=0;i<m;i++){ CK(cuMemUnmap(base+i*g,g)); CK(cuMemMap(base+i*g,g,0,dummy,0)); }
  CK(cuMemSetAccess(base,m*g,&ad,1)); t1=now();
  printf("swap back %.1f ms, avail %ld (delta %ld)\n",t1-t0,avail_mb(),avail_mb()-a0);
  t0=now(); for(size_t i=0;i<n;i++) CK(cuMemUnmap(base+i*g,g)); t1=now();
  printf("unmap all %.1f ms\n",t1-t0);
  CK(cuMemRelease(dummy)); CK(cuMemAddressFree(base,VA)); return 0; }
