// Read-only GLM K3XS benchmark. Persistent pread workers; optional liburing per worker.
#include "expert_disk.h"
#include <atomic>
#include <iostream>
#include <random>
using namespace std;
int main(int argc, char **argv) {
 try {
    if(argc != 6 && argc != 7) throw runtime_error("usage: glm_readbench STORE PIECE QD RECORDS SAMPLES [SECONDS]");
    k3disk::Store store(argv[1]);
    size_t piece=stoull(argv[2]); unsigned qd=stoul(argv[3]), records=stoul(argv[4]), samples=stoul(argv[5]);
    double seconds=argc==7?stod(argv[6]):4.0;
    if(!piece || piece%4096 || !qd || qd>16 || (records!=1 && records!=8) || !samples) throw runtime_error("bad args");
    vector<k3disk::Record> eligible;
    for(auto r: store.records) if(r.bytes==12091392) eligible.push_back(r);
    if(store.experts!=256 || eligible.size()%256 || eligible.size()<256) throw runtime_error("no GLM 12091392-byte records");
    int fd=open(argv[1],O_RDONLY|O_DIRECT); if(fd<0) throw runtime_error("O_DIRECT open failed");
    mutex m; condition_variable cv,done; bool stop=false; unsigned generation=0, finished=0;
    struct Job {uint64_t off; size_t bytes;}; vector<Job> jobs; atomic<size_t> next{0}; atomic<bool> failed{false}; atomic<unsigned> uring_workers{0};
    vector<thread> workers;
    for(unsigned w=0;w<qd;w++) workers.emplace_back([&]{
      void *buf=nullptr; if(posix_memalign(&buf,4096,piece)) {std::abort();}
#ifdef K3X_HAVE_URING
      io_uring ring{}; bool uring=io_uring_queue_init(1,&ring,0)==0; if(uring) ++uring_workers;
#endif
      unsigned seen=0; unique_lock<mutex> lock(m);
      while(true){cv.wait(lock,[&]{return stop || generation!=seen;}); if(stop) break; seen=generation; lock.unlock();
        for(size_t i;(i=next.fetch_add(1))<jobs.size();){auto j=jobs[i]; ssize_t n;
#ifdef K3X_HAVE_URING
          if(uring){auto *sqe=io_uring_get_sqe(&ring); io_uring_prep_read(sqe,fd,buf,j.bytes,j.off);
            int rc=io_uring_submit(&ring); io_uring_cqe *cqe=nullptr;
            if(rc==1){do{rc=io_uring_wait_cqe(&ring,&cqe);}while(rc==-EINTR);}else rc=-EIO;
            n=rc<0?rc:cqe->res; if(cqe) io_uring_cqe_seen(&ring,cqe);
          }else
#endif
          {do{n=pread(fd,buf,j.bytes,j.off);}while(n<0 && errno==EINTR);}
          if(n!=(ssize_t)j.bytes) failed=true;
        }
        lock.lock(); ++finished; done.notify_one();
      }
#ifdef K3X_HAVE_URING
      if(uring) io_uring_queue_exit(&ring);
#endif
      free(buf);
    });
    cout << unitbuf;
    mt19937 rng(530); bool bad=false;
    auto case_start=chrono::steady_clock::now();
    for(unsigned it=0;it<samples+5;it++){
      jobs.clear(); size_t base=rng()%(eligible.size()/256)*256; // one layer, eight distinct experts
      vector<unsigned> ids(256); for(unsigned i=0;i<256;i++)ids[i]=i; shuffle(ids.begin(),ids.end(),rng);
      for(unsigned r=0;r<records;r++){auto rec=eligible.at(base+ids[r]);
        for(size_t off=0;off<rec.bytes;off+=piece)jobs.push_back({rec.offset+off,min(piece,(size_t)rec.bytes-off)});
      }
      unique_lock<mutex> lock(m); next=0; finished=0; auto start=chrono::steady_clock::now(); ++generation; cv.notify_all();
      done.wait(lock,[&]{return finished==qd;});
      double us=chrono::duration<double,micro>(chrono::steady_clock::now()-start).count();
      if(failed){bad=true;break;} if(it>=5) cout<<"{\"hop\":\"nvme/records="<<records<<"/piece="<<piece<<"/qd="<<qd<<"\",\"us\":"<<us<<"}\n";
      if(chrono::steady_clock::now()-case_start > chrono::duration<double>(seconds)) break;
    }
    {lock_guard<mutex> lock(m);stop=true;cv.notify_all();} for(auto &t:workers)t.join(); close(fd);
    cout << "{\"hop\":\"nvme/backend\",\"uring_workers\":" << uring_workers << ",\"pread_workers\":" << qd-uring_workers << "}\n";
    if(bad) throw runtime_error("short/failed direct read");
 }catch(const exception &e){cerr<<e.what()<<'\n';return 1;}
}
