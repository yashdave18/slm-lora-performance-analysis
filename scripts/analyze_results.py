"""Validate recorded artifacts and reproduce publication figures/tables (no GPU)."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'baseline':'Base', 'lora-r8-long':'Rank 8 / 1e-4',
          'lora-r16-long':'Rank 16 / 1e-4', 'lora-r16-lr5e4':'Rank 16 / 5e-4'}
SHORT = {'Salesforce/wikitext':'WikiText', 'roneneldan/TinyStories':'TinyStories',
'fancyzhx/ag_news':'AG News', 'stanfordnlp/imdb':'IMDb', 'Yelp/yelp_review_full':'Yelp',
'fancyzhx/amazon_polarity':'Amazon polarity', 'abisee/cnn_dailymail':'CNN/DailyMail',
'EdinburghNLP/xsum':'XSum','rajpurkar/squad':'SQuAD contexts','dair-ai/emotion':'Emotion',
'cornell-movie-review-data/rotten_tomatoes':'Rotten Tomatoes'}


def read(path): return json.loads(path.read_text())
def csv_write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b): assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8),(a,b)


def main(root):
    root=Path(root)
    test=read(root/'final_test/summary.json'); bench=read(root/'inference_benchmarks/summary.json')
    rows=bench['rows']; cfg=bench['specification']['settings']
    actual={(r['variant'],r['precision'],r['batch_size'],r['prompt_length']) for r in rows}
    expected=set(itertools.product(cfg['variants'],cfg['precisions'],cfg['batch_sizes'],cfg['prompt_lengths']))
    assert actual==expected and len(rows)==len(expected)==72
    assert all(r['status']=='ok' for r in rows)
    assert len(list((root/'inference_benchmarks/cases').glob('*.json')))==72
    for r in rows:
        case={k:r[k] for k in ('variant','precision','batch_size','prompt_length','new_tokens')}
        sha=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
        saved=read(root/f'inference_benchmarks/cases/{sha}.json')
        assert saved['summary']==r
        samples=saved['measurements']; assert len(samples)==cfg['measured_runs']==5
        assert all(s['generated_tokens']==r['batch_size']*r['new_tokens'] for s in samples)
        seconds=[s['seconds'] for s in samples]
        close(r['latency_mean_ms'],1000*statistics.mean(seconds))
        close(r['generated_tokens_per_sec'],sum(s['generated_tokens'] for s in samples)/sum(seconds))
        close(r['peak_allocated_mib'],max(s['peak_allocated_mib'] for s in samples))
    for name,result in test['results'].items():
        assert result==read(root/f'final_test/{name}.json')
        m=result['metrics']; close(math.exp(m['token_loss']),m['ppl'])
        close(m['token_loss']/math.log(2),m['bpt'])
        assert sum(v['n_tokens'] for v in result['by_dataset'].values())==m['n_tokens']==244760
        assert sum(v['n_sequences'] for v in result['by_dataset'].values())==m['n_sequences']==1629
        close(sum(v['token_loss']*v['n_tokens'] for v in result['by_dataset'].values())/m['n_tokens'],m['token_loss'])
    base=test['results']['baseline']; selected=test['results']['selected']
    reduction=100*(1-selected['metrics']['ppl']/base['metrics']['ppl'])
    close(reduction,test['ppl_reduction_percent'])
    figdir=root/'figures'; tables=root/'tables'
    figdir.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    overall=[{'model':name,**{k:v for k,v in result['metrics'].items() if k!='ppl_by_length_bucket'}} for name,result in test['results'].items()]
    datasets=[]
    for name,b in base['by_dataset'].items():
        s=selected['by_dataset'][name]
        datasets.append({'dataset':name,'n_tokens':b['n_tokens'],'baseline_ppl':b['ppl'],'selected_ppl':s['ppl'],
                         'ppl_reduction_percent':100*(1-s['ppl']/b['ppl'])})
    validation=[{'variant':name,**info['validation_selection']} for name,info in bench['specification']['variants'].items() if name!='baseline']
    buckets=[{'length_bucket':k,'n_tokens':v['n_tokens'],'baseline_ppl':v['ppl'],
              'selected_ppl':selected['metrics']['ppl_by_length_bucket'][k]['ppl']} for k,v in base['metrics']['ppl_by_length_bucket'].items()]
    csv_write(tables/'test_overall.csv',overall);csv_write(tables/'test_by_dataset.csv',datasets)
    csv_write(tables/'validation_selection.csv',validation);csv_write(tables/'test_by_length.csv',buckets)
    csv_write(tables/'benchmark_all.csv',rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':200})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/f'{name}.pdf',bbox_inches='tight');fig.savefig(figdir/f'{name}.png',bbox_inches='tight');plt.close(fig)
    ordered=sorted(datasets,key=lambda r:r['ppl_reduction_percent'])
    fig,ax=plt.subplots(figsize=(7.6,4.3));y=np.arange(len(ordered))
    ax.barh(y,[r['ppl_reduction_percent'] for r in ordered],color='#167d9a')
    ax.set_yticks(y,[SHORT[r['dataset']] for r in ordered]);ax.set_xlim(0,70)
    for i,r in enumerate(ordered):ax.text(r['ppl_reduction_percent']+0.7,i,f"{r['ppl_reduction_percent']:.1f}%",va='center',fontsize=9)
    ax.set_xlabel('Test perplexity reduction relative to base (%)');ax.set_title('Selected adapter improves all 11 sources')
    save(fig,'test_by_dataset')
    fig,ax=plt.subplots(figsize=(7.6,2.7))
    ax.bar([LABELS[r['variant']] for r in validation],[r['ppl'] for r in validation],color=['#879ba4','#467b9e','#167d9a'])
    ax.set_ylim(0,22);ax.set_ylabel('Validation perplexity');ax.set_title('Validation-selected checkpoints (lower is better)')
    for i,r in enumerate(validation):ax.text(i,r['ppl']+.25,f"{r['ppl']:.4f}",ha='center')
    save(fig,'validation_comparison')
    variants=cfg['variants'];colors=['#596773','#de9239','#5589aa','#167d9a']
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for ax,precision in zip(axes,['fp32','fp16']):
        for name,color in zip(variants,colors):
            rr=sorted([r for r in rows if r['variant']==name and r['precision']==precision and r['prompt_length']==128],key=lambda x:x['batch_size'])
            ax.plot([r['batch_size'] for r in rr],[r['generated_tokens_per_sec'] for r in rr],'-o',color=color,label=LABELS[name])
        ax.set_title('FP32 compute' if precision=='fp32' else 'FP16 autocast');ax.set_xticks([1,2,4]);ax.set_xlabel('Batch size');ax.grid(alpha=.15)
    axes[0].set_ylabel('Generated tokens / second (batch aggregate)');axes[1].legend(fontsize=8)
    fig.suptitle('128-token prompts; 64 new tokens; unmerged adapters',fontsize=11)
    save(fig,'throughput_by_batch')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3));x=np.arange(4);width=.36
    for j,(precision,color) in enumerate([('fp32','#167d9a'),('fp16','#de9239')]):
        rr=[next(r for r in rows if r['variant']==n and r['precision']==precision and r['batch_size']==1 and r['prompt_length']==128) for n in variants]
        axes[0].bar(x+(j-.5)*width,[r['latency_mean_ms'] for r in rr],width,yerr=[r['latency_stdev_ms'] for r in rr],capsize=3,color=color,label=precision)
        axes[1].bar(x+(j-.5)*width,[r['peak_allocated_mib'] for r in rr],width,color=color,label=precision)
    for ax in axes:ax.set_xticks(x,[LABELS[n] for n in variants],rotation=20,ha='right');ax.legend(fontsize=8)
    axes[0].set_ylabel('Full generation latency (ms)');axes[0].set_title('Mean +/- repeat SD (n=5)')
    axes[1].set_ylabel('Peak allocated GPU memory (MiB)');axes[1].set_title('FP32 stored weights in both modes')
    save(fig,'latency_memory')
    fig,axes=plt.subplots(1,2,figsize=(9,3.2))
    for precision,color in [('fp32','#167d9a'),('fp16','#de9239')]:
        rr=sorted([r for r in rows if r['variant']=='lora-r16-lr5e4' and r['precision']==precision and r['batch_size']==4],key=lambda r:r['prompt_length'])
        for ax,key in zip(axes,['generated_tokens_per_sec','peak_allocated_mib']):
            ax.plot([r['prompt_length'] for r in rr],[r[key] for r in rr],'-o',color=color,label=precision);ax.set_xticks([64,128,256]);ax.set_xlabel('Prompt tokens');ax.legend()
    axes[0].set_ylabel('Generated tokens / second');axes[1].set_ylabel('Peak allocated GPU memory (MiB)')
    fig.suptitle('Selected adapter, batch 4, 64 new tokens',fontsize=11);save(fig,'prompt_length')
    integrity={'benchmark_cases':len(rows),'measurements':len(rows)*5,'test_tokens':244760,'test_sequences':1629,
               'test_ppl_reduction_percent':reduction,'datasets_improved':sum(r['ppl_reduction_percent']>0 for r in datasets),
               'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for d in ('final_test','inference_benchmarks') for p in sorted((root/d).rglob('*')) if p.is_file()}}
    (tables/'verification.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print(f'PASS: {len(rows)} cases, 360 timed generations, 244760 test targets verified.')
    print(f'Test perplexity: {base["metrics"]["ppl"]:.6f} -> {selected["metrics"]["ppl"]:.6f} ({reduction:.2f}% reduction)')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,default=Path('results'))
    main(p.parse_args().results_dir)
