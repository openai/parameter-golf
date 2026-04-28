
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
typedef struct{uint64_t key;uint32_t total,max_count,unique,head;uint8_t used,ib[4];uint32_t ic[4];} Ctx;
typedef struct{uint32_t next,ctx,count;uint8_t byte;} Edge;
typedef struct{Ctx*ctx;uint64_t cap,used;Edge*edges;uint64_t ecap,eused;} Table;
static uint64_t mix64(uint64_t x){x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;x*=0xc4ceb9fe1a85ec53ULL;x^=x>>33;return x;}
static int table_init(Table*t,uint64_t cap){uint64_t c=1;while(c<cap)c<<=1;t->cap=c;t->used=0;t->ctx=(Ctx*)calloc(c,sizeof(Ctx));t->ecap=cap*2+1024;t->eused=1;t->edges=(Edge*)calloc(t->ecap,sizeof(Edge));return t->ctx&&t->edges?0:-1;}
static void table_free(Table*t){free(t->ctx);free(t->edges);memset(t,0,sizeof(*t));}
static int grow_edges(Table*t){uint64_t nc=t->ecap*2;Edge*ne=(Edge*)realloc(t->edges,nc*sizeof(Edge));if(!ne)return-1;memset(ne+t->ecap,0,(nc-t->ecap)*sizeof(Edge));t->edges=ne;t->ecap=nc;return 0;}
static Ctx* table_find(Table*t,uint64_t key){uint64_t m=t->cap-1,i=mix64(key)&m;for(;;){Ctx*c=&t->ctx[i];if(!c->used)return 0;if(c->key==key)return c;i=(i+1)&m;}}
static int table_rehash(Table*t){
    Table nt;if(table_init(&nt,t->cap*2))return-1;
    free(nt.edges);nt.edges=t->edges;nt.ecap=t->ecap;nt.eused=t->eused;
    for(uint64_t j=0;j<t->cap;j++)if(t->ctx[j].used){uint64_t m=nt.cap-1,i=mix64(t->ctx[j].key)&m;while(nt.ctx[i].used)i=(i+1)&m;nt.ctx[i]=t->ctx[j];nt.used++;}
    free(t->ctx);*t=nt;return 0;
}
static Ctx* table_get_or_add(Table*t,uint64_t key){
    if((t->used+1)*10>t->cap*7)if(table_rehash(t))return 0;
    uint64_t m=t->cap-1,i=mix64(key)&m;
    for(;;){Ctx*c=&t->ctx[i];if(!c->used){c->used=1;c->key=key;c->head=0;t->used++;return c;}if(c->key==key)return c;i=(i+1)&m;}
}
static uint32_t edge_count(Table*t,Ctx*c,uint8_t b){uint32_t m=c->unique<4?c->unique:4;for(uint32_t i=0;i<m;i++)if(c->ib[i]==b)return c->ic[i];for(uint32_t e=c->head;e;e=t->edges[e].next)if(t->edges[e].byte==b)return t->edges[e].count;return 0;}
static int edge_inc(Table*t,Ctx*c,uint8_t b){
    uint32_t m=c->unique<4?c->unique:4;for(uint32_t i=0;i<m;i++)if(c->ib[i]==b){uint32_t nc=++c->ic[i];c->total++;if(nc>c->max_count)c->max_count=nc;return 0;}
    for(uint32_t e=c->head;e;e=t->edges[e].next)if(t->edges[e].byte==b){uint32_t nc=++t->edges[e].count;c->total++;if(nc>c->max_count)c->max_count=nc;return 0;}
    if(c->unique<4){uint32_t i=c->unique;c->ib[i]=b;c->ic[i]=1;c->total++;c->unique++;if(c->max_count<1)c->max_count=1;return 0;}
    if(t->eused>=t->ecap)if(grow_edges(t))return-1;
    uint32_t e=(uint32_t)t->eused++;t->edges[e].byte=b;t->edges[e].count=1;t->edges[e].ctx=(uint32_t)(c-t->ctx);t->edges[e].next=c->head;c->head=e;c->total++;c->unique++;if(c->max_count<1)c->max_count=1;return 0;
}
static uint64_t mask_for(int K){return K>=8?~0ULL:((1ULL<<(8*K))-1ULL);}
static inline double lgi(uint32_t x,double*lc,uint32_t lcap){if(lc&&x<lcap){double v=lc[x];if(v>=0.0)return v;v=log((double)x);lc[x]=v;return v;}return log((double)x);}

/* score_byte_with_dump: extension of score_byte that ALSO writes per-byte
 * info (mix_nll, ppm_nll, nn_nll, conf, gate_high, actual_byte) to dump
 * arrays at index *dump_idx, then increments dump_idx. If dump arrays are
 * NULL, behaves identically to score_byte. */
static int score_byte_with_dump(Table*tables,uint32_t*c0,uint32_t*tot0,uint32_t*uniq0,uint32_t*max0,uint64_t*hist,int*wlen,int order,uint8_t b,double nn_logp,double lambda_hi,double lambda_lo,double lhi,double llo,double l1hi,double l1lo,double thr,double*lc,uint32_t lcap,double*mix_nll,double*ppm_nll,double*nn_nll,uint64_t*bytes,uint64_t*gate_high,uint64_t*gate_total,
        float*dump_mix,float*dump_ppm,float*dump_nn,float*dump_conf,uint8_t*dump_gate_hi,uint8_t*dump_byte,uint64_t*dump_idx){
    const double uni=log(1.0/256.0);double ppm_log=0.0,conf=0.0,esc=0.0;int found=0,seen=0,maxk=*wlen<order?*wlen:order;uint64_t keys[9];keys[0]=0;for(int K=1;K<=maxk;K++)keys[K]=(*hist)&mask_for(K);
    for(int K=maxk;K>=1;K--){Ctx*c=table_find(&tables[K],keys[K]);if(!c)continue;uint32_t den=c->total+c->unique;if(!den)continue;double denom=(double)den;if(!seen){conf=(double)c->max_count/denom;seen=1;}uint32_t cnt=edge_count(&tables[K],c,b);if(cnt){ppm_log=esc+(lgi(cnt,lc,lcap)-lgi(den,lc,lcap));found=1;break;}if(c->unique>0)esc+=lgi(c->unique,lc,lcap)-lgi(den,lc,lcap);}
    if(!found){uint32_t den0=*tot0+*uniq0;if(den0>0){double denom0=(double)den0;if(!seen){conf=(double)(*max0)/denom0;seen=1;}uint32_t cnt=c0[b];if(cnt){ppm_log=esc+(lgi(cnt,lc,lcap)-lgi(den0,lc,lcap));found=1;}else if(*uniq0>0)esc+=lgi(*uniq0,lc,lcap)-lgi(den0,lc,lcap);}}
    if(!found)ppm_log=esc+uni;
    int hi=conf>=thr;double lam=hi?lambda_lo:lambda_hi;(*gate_total)++;if(hi)(*gate_high)++;
    double log_mix;if(lam<=0.0)log_mix=ppm_log;else if(lam>=1.0)log_mix=nn_logp;else{double a=(hi?llo:lhi)+nn_logp,c=(hi?l1lo:l1hi)+ppm_log,m=a>c?a:c;log_mix=m+log(exp(a-m)+exp(c-m));}
    *mix_nll-=log_mix;*ppm_nll-=ppm_log;*nn_nll-=nn_logp;(*bytes)++;
    /* PER-BYTE DUMP */
    if(dump_mix){
        uint64_t idx=*dump_idx;
        dump_mix[idx]=(float)(-log_mix);
        dump_ppm[idx]=(float)(-ppm_log);
        dump_nn[idx]=(float)(-nn_logp);
        dump_conf[idx]=(float)conf;
        dump_gate_hi[idx]=(uint8_t)hi;
        dump_byte[idx]=b;
        (*dump_idx)++;
    }
    uint32_t nc=++c0[b];(*tot0)++;if(nc==1)(*uniq0)++;if(nc>*max0)*max0=nc;
    for(int K=1;K<=maxk;K++){Ctx*c=table_get_or_add(&tables[K],keys[K]);if(!c||edge_inc(&tables[K],c,b))return-1;}
    if(order>0){*hist=((*hist)<<8|b)&mask_for(order);if(*wlen<order)(*wlen)++;}
    return 0;
}

/* Backward-compat wrapper: calls score_byte_with_dump with NULL dump pointers. */
static int score_byte(Table*tables,uint32_t*c0,uint32_t*tot0,uint32_t*uniq0,uint32_t*max0,uint64_t*hist,int*wlen,int order,uint8_t b,double nn_logp,double lambda_hi,double lambda_lo,double lhi,double llo,double l1hi,double l1lo,double thr,double*lc,uint32_t lcap,double*mix_nll,double*ppm_nll,double*nn_nll,uint64_t*bytes,uint64_t*gate_high,uint64_t*gate_total){
    return score_byte_with_dump(tables,c0,tot0,uniq0,max0,hist,wlen,order,b,nn_logp,lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,lcap,mix_nll,ppm_nll,nn_nll,bytes,gate_high,gate_total,
        NULL,NULL,NULL,NULL,NULL,NULL,NULL);
}

int ppm_score(const int64_t*target,const int64_t*prev,const double*nll,int64_t n,const uint8_t*flat,const int32_t*offs,const int32_t*lens,const uint8_t*has_space,const uint8_t*is_boundary,int vocab,int order,double lambda_hi,double lambda_lo,double thr,uint32_t log_cache_size,double*out){
    if(order<0||order>8)return-2;Table tables[9];uint64_t cap=(uint64_t)n*2+1024;for(int k=1;k<=order;k++)if(table_init(&tables[k],cap/(k+1)+1024))return-3;
    double*lc=0;if(log_cache_size>1){lc=(double*)malloc((size_t)log_cache_size*sizeof(double));if(!lc)return-6;for(uint32_t i=0;i<log_cache_size;i++)lc[i]=-1.0;}double lhi=log(lambda_hi),llo=log(lambda_lo),l1hi=log(1.0-lambda_hi),l1lo=log(1.0-lambda_lo);
    uint32_t c0[256];memset(c0,0,sizeof(c0));uint32_t tot0=0,uniq0=0,max0=0;uint64_t hist=0;int wlen=0;double mix_nll=0,ppm_nll=0,nn_nll=0,token_nll=0;uint64_t bytes=0,gate_high=0,gate_total=0;
    for(int64_t i=0;i<n;i++){int tid=(int)target[i],pid=(int)prev[i];if(tid<0||tid>=vocab)continue;int len=lens[tid];int inc_space=has_space[tid]&&(pid<0||!is_boundary[pid]);int nb=len+(inc_space?1:0);if(nb<=0)continue;double nn_logp=-nll[i]/(double)nb;token_nll+=nll[i];if(inc_space)if(score_byte(tables,c0,&tot0,&uniq0,&max0,&hist,&wlen,order,32,nn_logp,lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,log_cache_size,&mix_nll,&ppm_nll,&nn_nll,&bytes,&gate_high,&gate_total))return-4;const uint8_t*p=flat+offs[tid];for(int j=0;j<len;j++)if(score_byte(tables,c0,&tot0,&uniq0,&max0,&hist,&wlen,order,p[j],nn_logp,lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,log_cache_size,&mix_nll,&ppm_nll,&nn_nll,&bytes,&gate_high,&gate_total))return-5;}
    const double log2v=log(2.0);out[0]=bytes?mix_nll/(double)bytes/log2v:0;out[1]=bytes?ppm_nll/(double)bytes/log2v:0;out[2]=bytes?nn_nll/(double)bytes/log2v:0;out[3]=bytes?token_nll/(double)bytes/log2v:0;out[4]=(double)bytes;out[5]=gate_total?(double)gate_high/(double)gate_total:0;
    if(lc)free(lc);for(int k=1;k<=order;k++)table_free(&tables[k]);return 0;
}

/* ppm_score_bytewise: takes a flat byte stream + per-byte nn_logp array
 * (proper marginalization log-prob from external Python walker). Loops
 * byte-by-byte and applies PPM-D mix. No token-based reconstruction.
 * Returns out[0..5] with same layout as ppm_score, but out[3] (token_nll)
 * is set to 0 (no token info here). */
int ppm_score_bytewise(const uint8_t*byte_stream,const double*nn_logp_per_byte,int64_t n_bytes,
        int order,double lambda_hi,double lambda_lo,double thr,uint32_t log_cache_size,double*out){
    if(order<0||order>8)return-2;
    Table tables[9];uint64_t cap=(uint64_t)n_bytes*2+1024;
    for(int k=1;k<=order;k++)if(table_init(&tables[k],cap/(k+1)+1024))return-3;
    double*lc=0;if(log_cache_size>1){lc=(double*)malloc((size_t)log_cache_size*sizeof(double));if(!lc)return-6;for(uint32_t i=0;i<log_cache_size;i++)lc[i]=-1.0;}
    double lhi=log(lambda_hi),llo=log(lambda_lo),l1hi=log(1.0-lambda_hi),l1lo=log(1.0-lambda_lo);
    uint32_t c0[256];memset(c0,0,sizeof(c0));uint32_t tot0=0,uniq0=0,max0=0;uint64_t hist=0;int wlen=0;
    double mix_nll=0,ppm_nll=0,nn_nll=0;uint64_t bytes=0,gate_high=0,gate_total=0;
    for(int64_t i=0;i<n_bytes;i++){
        uint8_t b=byte_stream[i];
        double nn_logp=nn_logp_per_byte[i];
        if(score_byte(tables,c0,&tot0,&uniq0,&max0,&hist,&wlen,order,b,nn_logp,
                lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,log_cache_size,
                &mix_nll,&ppm_nll,&nn_nll,&bytes,&gate_high,&gate_total))return-5;
    }
    const double log2v=log(2.0);
    out[0]=bytes?mix_nll/(double)bytes/log2v:0;
    out[1]=bytes?ppm_nll/(double)bytes/log2v:0;
    out[2]=bytes?nn_nll/(double)bytes/log2v:0;
    out[3]=0;out[4]=(double)bytes;out[5]=gate_total?(double)gate_high/(double)gate_total:0;
    if(lc)free(lc);for(int k=1;k<=order;k++)table_free(&tables[k]);return 0;
}

/* ppm_score_dump: single-threaded ppm_score with per-byte data dump.
 * Caller pre-allocates dump arrays of size >= max possible byte count
 * (suggest: n_tokens * 17 to be safe). On return, `*dump_n_bytes`
 * contains actual bytes written. Other behavior identical to ppm_score. */
int ppm_score_dump(const int64_t*target,const int64_t*prev,const double*nll,int64_t n,const uint8_t*flat,const int32_t*offs,const int32_t*lens,const uint8_t*has_space,const uint8_t*is_boundary,int vocab,int order,double lambda_hi,double lambda_lo,double thr,uint32_t log_cache_size,double*out,
        float*dump_mix,float*dump_ppm,float*dump_nn,float*dump_conf,uint8_t*dump_gate_hi,uint8_t*dump_byte,uint64_t*dump_n_bytes){
    if(order<0||order>8)return-2;Table tables[9];uint64_t cap=(uint64_t)n*2+1024;for(int k=1;k<=order;k++)if(table_init(&tables[k],cap/(k+1)+1024))return-3;
    double*lc=0;if(log_cache_size>1){lc=(double*)malloc((size_t)log_cache_size*sizeof(double));if(!lc)return-6;for(uint32_t i=0;i<log_cache_size;i++)lc[i]=-1.0;}double lhi=log(lambda_hi),llo=log(lambda_lo),l1hi=log(1.0-lambda_hi),l1lo=log(1.0-lambda_lo);
    uint32_t c0[256];memset(c0,0,sizeof(c0));uint32_t tot0=0,uniq0=0,max0=0;uint64_t hist=0;int wlen=0;double mix_nll=0,ppm_nll=0,nn_nll=0,token_nll=0;uint64_t bytes=0,gate_high=0,gate_total=0;
    uint64_t dump_idx=0;
    for(int64_t i=0;i<n;i++){
        int tid=(int)target[i],pid=(int)prev[i];if(tid<0||tid>=vocab)continue;
        int len=lens[tid];int inc_space=has_space[tid]&&(pid<0||!is_boundary[pid]);int nb=len+(inc_space?1:0);if(nb<=0)continue;
        double nn_logp=-nll[i]/(double)nb;token_nll+=nll[i];
        if(inc_space)if(score_byte_with_dump(tables,c0,&tot0,&uniq0,&max0,&hist,&wlen,order,32,nn_logp,lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,log_cache_size,&mix_nll,&ppm_nll,&nn_nll,&bytes,&gate_high,&gate_total,
            dump_mix,dump_ppm,dump_nn,dump_conf,dump_gate_hi,dump_byte,&dump_idx))return-4;
        const uint8_t*p=flat+offs[tid];
        for(int j=0;j<len;j++)if(score_byte_with_dump(tables,c0,&tot0,&uniq0,&max0,&hist,&wlen,order,p[j],nn_logp,lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,log_cache_size,&mix_nll,&ppm_nll,&nn_nll,&bytes,&gate_high,&gate_total,
            dump_mix,dump_ppm,dump_nn,dump_conf,dump_gate_hi,dump_byte,&dump_idx))return-5;
    }
    const double log2v=log(2.0);
    out[0]=bytes?mix_nll/(double)bytes/log2v:0;out[1]=bytes?ppm_nll/(double)bytes/log2v:0;out[2]=bytes?nn_nll/(double)bytes/log2v:0;out[3]=bytes?token_nll/(double)bytes/log2v:0;out[4]=(double)bytes;out[5]=gate_total?(double)gate_high/(double)gate_total:0;
    *dump_n_bytes=dump_idx;
    if(lc)free(lc);for(int k=1;k<=order;k++)table_free(&tables[k]);return 0;
}

/* OpenMP parallel chunked scorer. Splits the token stream into chunks of
 * size `chunk_tokens`; each chunk gets its own PPM-D state (tables, c0,
 * hist) and is processed sequentially within the chunk. Chunks are
 * distributed across OMP threads via dynamic scheduling. PPM state
 * RESETS at chunk boundaries -- this CHANGES the scored BPB vs the
 * single-context legacy ppm_score path (smaller chunks => more cold-start).
 * For chunk_tokens >= n the result is bit-identical to ppm_score.
 * `lc` (log-cache) is a read-only memo populated lazily; with
 * `lc[i] = log(i)` it is monotonically writable -- benign races are
 * idempotent (every thread writes the same value). To be conservative
 * each thread allocates its own log cache. */
int ppm_score_omp(const int64_t*target,const int64_t*prev,const double*nll,int64_t n,const uint8_t*flat,const int32_t*offs,const int32_t*lens,const uint8_t*has_space,const uint8_t*is_boundary,int vocab,int order,double lambda_hi,double lambda_lo,double thr,uint32_t log_cache_size,int64_t chunk_tokens,int num_threads,double*out){
    if(order<0||order>8)return-2;
    if(chunk_tokens<=0)return-7;
    if(num_threads>0)omp_set_num_threads(num_threads);
    double lhi=log(lambda_hi),llo=log(lambda_lo),l1hi=log(1.0-lambda_hi),l1lo=log(1.0-lambda_lo);
    int64_t num_chunks=(n+chunk_tokens-1)/chunk_tokens;
    double mix_nll_total=0,ppm_nll_total=0,nn_nll_total=0,token_nll_total=0;
    uint64_t bytes_total=0,gate_high_total=0,gate_total_total=0;
    int err_code=0;
    #pragma omp parallel for schedule(dynamic,1) reduction(+:mix_nll_total,ppm_nll_total,nn_nll_total,token_nll_total,bytes_total,gate_high_total,gate_total_total)
    for(int64_t ci=0;ci<num_chunks;ci++){
        if(err_code)continue;
        int64_t s=ci*chunk_tokens;
        int64_t e=s+chunk_tokens;
        if(e>n)e=n;
        int64_t cn=e-s;
        Table tables[9];memset(tables,0,sizeof(tables));
        uint64_t cap=(uint64_t)cn*2+1024;
        int local_err=0;
        for(int k=1;k<=order;k++)if(table_init(&tables[k],cap/(k+1)+1024)){local_err=-3;break;}
        double*lc=0;
        if(!local_err&&log_cache_size>1){lc=(double*)malloc((size_t)log_cache_size*sizeof(double));if(!lc)local_err=-6;else for(uint32_t i=0;i<log_cache_size;i++)lc[i]=-1.0;}
        if(!local_err){
            uint32_t c0[256];memset(c0,0,sizeof(c0));
            uint32_t tot0=0,uniq0=0,max0=0;
            uint64_t hist=0;int wlen=0;
            double mix_nll=0,ppm_nll=0,nn_nll=0,token_nll=0;
            uint64_t bytes=0,gate_high=0,gate_total=0;
            for(int64_t i=s;i<e&&!local_err;i++){
                int tid=(int)target[i],pid=(int)prev[i];
                if(tid<0||tid>=vocab)continue;
                int len=lens[tid];
                int inc_space=has_space[tid]&&(pid<0||!is_boundary[pid]);
                int nb=len+(inc_space?1:0);
                if(nb<=0)continue;
                double nn_logp=-nll[i]/(double)nb;
                token_nll+=nll[i];
                if(inc_space)if(score_byte(tables,c0,&tot0,&uniq0,&max0,&hist,&wlen,order,32,nn_logp,lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,log_cache_size,&mix_nll,&ppm_nll,&nn_nll,&bytes,&gate_high,&gate_total)){local_err=-4;break;}
                const uint8_t*p=flat+offs[tid];
                for(int j=0;j<len;j++)if(score_byte(tables,c0,&tot0,&uniq0,&max0,&hist,&wlen,order,p[j],nn_logp,lambda_hi,lambda_lo,lhi,llo,l1hi,l1lo,thr,lc,log_cache_size,&mix_nll,&ppm_nll,&nn_nll,&bytes,&gate_high,&gate_total)){local_err=-5;break;}
            }
            if(!local_err){
                mix_nll_total+=mix_nll;ppm_nll_total+=ppm_nll;nn_nll_total+=nn_nll;token_nll_total+=token_nll;
                bytes_total+=bytes;gate_high_total+=gate_high;gate_total_total+=gate_total;
            }
        }
        if(lc)free(lc);
        for(int k=1;k<=order;k++)if(tables[k].ctx)table_free(&tables[k]);
        if(local_err){
            #pragma omp atomic write
            err_code=local_err;
        }
    }
    if(err_code)return err_code;
    const double log2v=log(2.0);
    out[0]=bytes_total?mix_nll_total/(double)bytes_total/log2v:0;
    out[1]=bytes_total?ppm_nll_total/(double)bytes_total/log2v:0;
    out[2]=bytes_total?nn_nll_total/(double)bytes_total/log2v:0;
    out[3]=bytes_total?token_nll_total/(double)bytes_total/log2v:0;
    out[4]=(double)bytes_total;
    out[5]=gate_total_total?(double)gate_high_total/(double)gate_total_total:0;
    return 0;
}
