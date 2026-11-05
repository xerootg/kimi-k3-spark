// GH200: does MADV_DONTNEED on cudaHostRegister'd, mlocked memory bound to the HBM NUMA node give the memory back,
// and does the GPU see what the CPU writes into the refaulted pages afterwards?
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
static long node_free_mb(int n){ char p[96]; snprintf(p,96,"/sys/devices/system/node/node%d/meminfo",n); FILE*f=fopen(p,"r"); char l[256]; long v=-1;
  while(fgets(l,256,f)){ char*q=strstr(l,"MemFree:"); if(q){ sscanf(q+8,"%ld",&v); break; } } fclose(f); return v/1024; }
__global__ void rd(const unsigned char*p, size_t n, unsigned long long*out){ unsigned long long s=0; for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x*4096/4096) { if(i%4096==0) s+=p[i]; } atomicAdd(out,s); }
int main(int argc,char**argv){ int node=argc>1?atoi(argv[1]):1; const size_t N=(size_t)4<<30, H=(size_t)2<<30;
  cudaFree(0);
  unsigned char*p=(unsigned char*)mmap(0,N,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  unsigned long mask=1UL<<node; if(syscall(SYS_mbind,p,N,2/*MPOL_BIND*/,&mask,64,0)) { perror("mbind"); return 1; }
  long f0=node_free_mb(node); memset(p,1,N); long f1=node_free_mb(node);
  printf("node %d free: before touch %ld MiB, after touching 4 GiB %ld (delta %ld)\n",node,f0,f1,f1-f0);
  if(mlock(p,N)) perror("mlock");
  cudaError_t e=cudaHostRegister(p,N,cudaHostRegisterPortable); printf("cudaHostRegister: %s\n",cudaGetErrorString(e));
  unsigned long long*d; cudaMallocManaged(&d,8); *d=0; rd<<<264,256>>>(p,H,d); cudaDeviceSynchronize(); printf("GPU sum over first 2 GiB (1 per page expected %zu): %llu\n",H/4096,*d);
  munlock(p,H); int r=madvise(p,H,MADV_DONTNEED); long f2=node_free_mb(node); printf("munlock+MADV_DONTNEED 2 GiB -> %d, node free %ld (delta %ld)\n",r,f2,f2-f1);
  memset(p,2,H); long f3=node_free_mb(node); printf("CPU rewrote 2 GiB with 2s: node free %ld (delta vs after-touch %ld)\n",f3,f3-f1);
  *d=0; rd<<<264,256>>>(p,H,d); cudaDeviceSynchronize(); printf("GPU sum after rewrite (2 per page = %zu if the GPU sees the new pages, %zu if stale): %llu\n",2*(H/4096),H/4096,*d);
  cudaHostUnregister(p); long f4=node_free_mb(node); munmap(p,N); long f5=node_free_mb(node); printf("after unregister %ld, after munmap %ld\n",f4,f5); return 0; }
