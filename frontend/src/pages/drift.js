import {esc,localTime,fmt} from '../utils/format.js';
import {domain} from '../constants/domain.js';
import {research} from '../stores/research.js';
import {session} from '../stores/session.js';
import {driftApi,runsApi} from '../api/research.js';
import {status} from '../components/common/status.js';
import {table} from '../components/common/table.js';
import {runOptions} from '../components/common/business.js';
import {action,toast,bind} from '../hooks/actions.js';

const jobStates={queued:'队列中',running:'正在计算',completed:'已完成',failed:'失败'};

function planLedger(plans){
  if(!plans.length)return '<div class="empty">还没有归班版本。先在下方为已完成研究补录样本归班与班次顺序。</div>';
  return '<div class="ledger">'+plans.map(p=>'<div><small>r'+p.revision+'</small><strong>'+esc(p.created_by)+'</strong><span>'+p.shift_count+' 班 · '+p.reference_count+' 件</span><time title="北京时间">'+esc(localTime(p.created_at))+'</time></div>').join('')+'</div>';
}

function planForm(selected){
  return '<form id="plan-form"><label>来源批次<select name="source_id">'+(research.sources.map(s=>'<option value="'+s.id+'" '+(s.id==selected?'selected':'')+'>'+esc(s.name)+' · '+s.record_count+'条</option>').join(''))+'</select></label><label>版本说明<input name="note" value="'+esc(domain.source)+'归班补录" maxlength="2000"></label><label>归班明细<textarea name="assignments" rows="8" required placeholder="班次名称,班次顺序,样本编号&#10;白班,1,TX-202611-01-0001&#10;中班,2,TX-202611-01-0002"></textarea></label><p class="fine">每行一件样本：班次名称,班次顺序,样本编号。更正归班会另起一个版本，旧版本与其计算保持冻结。</p><button type="submit">保存归班新版本 →</button></form>';
}

function parseAssignments(text){
  const shifts=new Map();
  for(const line of text.split(/\r?\n/)){
    const raw=line.trim();if(!raw)continue;
    const parts=raw.split(',');
    if(parts.length<3)throw Error('归班明细需要三列：班次名称,班次顺序,样本编号');
    const name=parts[0].trim(),order=Number(parts[1].trim());
    if(!name)throw Error('班次名称不能为空');
    if(!Number.isInteger(order)||order<1)throw Error('班次顺序需为正整数：'+raw);
    const key=name+'@'+order;
    if(!shifts.has(key))shifts.set(key,{name,order,references:[]});
    for(const ref of parts.slice(2))if(ref.trim())shifts.get(key).references.push(ref.trim());
  }
  if(!shifts.size)throw Error('请至少填写一件归班样本');
  return [...shifts.values()];
}

function segmentTable(result){
  return '<h2>漂移分界 <small>按指定故障的频谱质心 · 罚分 '+fmt(result.penalty)+'</small></h2><p class="fine">总分 '+fmt(result.total)+' = 绝对差 '+fmt(result.deviation_total)+' + 分界罚分 '+fmt(result.penalty_total)+' · 共 '+result.shift_total+' 班 / '+result.sample_total+' 件 · 每段至少 '+result.min_shifts+' 班</p>'+table(result.segments.map((s,i)=>({index:'第 '+(i+1)+' 段',shifts:s.shift_names[0]+' … '+s.shift_names[s.shift_names.length-1],shift_count:s.shift_count,samples:s.sample_count,median:s.median,deviation:s.deviation,after:i<result.segments.length-1?'分界 → '+s.shift_names[s.shift_names.length-1]:'—'})),[{key:'index',label:'段'},{key:'shifts',label:'覆盖班次'},{key:'shift_count',label:'班次数'},{key:'samples',label:'样本数'},{key:'median',label:'本段中位数 Hz'},{key:'deviation',label:'到中位数绝对差'},{key:'after',label:'分界'}]);
}

export async function driftPage(){
  const plans=await driftApi.plans(research.sourceId);
  const jobs=await driftApi.jobs();
  const results=await driftApi.results(research.sourceId);
  const latest=results[0];
  const selected=jobs.find(j=>j.id===research.driftJobId)||jobs[0];
  let jobDetail=null,labels=null;
  if(selected)jobDetail=await driftApi.job(selected.id);
  const completedRuns=research.runs.filter(r=>r.state==='completed');
  const runId=research.driftRunId||completedRuns[0]?.id;
  if(runId)labels=await driftApi.labels(runId);
  const job=jobDetail;
  return '<section class="page-heading"><p class="eyebrow">07 / DRIFT BOUNDARY</p><h1>同类故障的声纹，从哪一班开始漂。</h1><p>为已完成研究补录样本归班与班次顺序，按指定故障的频谱质心求最低总分的分界。每段至少三班；总分是各样本到本段中位数的绝对差与每道分界的固定罚分之和。</p></section><div class="split"><section>'+(latest?'<h2>研究页面 · 已发布分界 <small>'+esc(latest.label)+' · r'+latest.plan_revision+' · 版本 '+latest.version+'</small></h2>'+segmentTable(latest.result):'<div class="empty">该来源还没有已发布的分界。</div>')+'<h2>归班版本 <small>按来源</small></h2>'+planLedger(plans)+(session.writable?'<details open>'+planForm(research.sourceId)+'</details>':'')+'</section><aside class="entry-sheet"><h2>计算漂移分界</h2>'+(session.writable?'<form id="drift-form"><label>已完成研究<select name="run_id">'+runOptions(completedRuns,runId,true)+'</select></label><label>指定故障<select name="label">'+((labels?.labels||[]).map(l=>'<option value="'+esc(l.label)+'">'+esc(l.label)+' · '+l.count+' 件'+(l.unassigned?' · 未归班 '+l.unassigned:'')+'</option>').join('')||'<option value="">暂无已完成研究</option>')+'</select></label><label>每道分界固定罚分<input name="penalty" type="number" step="any" min="0" required value="1000"></label><button type="submit">进入计算队列 →</button></form>':'<p>阅读者可以查看已发布分界；归班与计算由分析师或负责人执行。</p>')+'<h2>计算任务</h2>'+(jobs.length?'<div class="ledger">'+jobs.map(j=>'<a href="#drift" data-drift-job="'+j.id+'"><small>#'+j.id+'</small><strong>'+esc(j.label)+' · r'+j.plan_revision+'</strong>'+status(j.state)+'<time>'+esc(localTime(j.created_at))+'</time></a>').join('')+'</div>':'<p class="fine">还没有漂移任务。</p>')+'</aside></div>'+(job?'<section class="drift-detail"><div class="run-title"><h2>任务 #'+job.id+' · '+esc(job.label)+'</h2>'+status(job.state)+'<small>研究 #'+job.run_id+' · 归班 r'+job.plan_revision+' · 罚分 '+fmt(job.parameters.penalty)+'</small>'+(job.state==='completed'&&session.writable?'<button id="drift-publish" class="secondary">重新发布 ↗</button>':'')+((job.state==='failed'||job.state==='running')&&session.writable?'<button id="drift-retry" class="secondary">续跑 / 重试 ↗</button>':'')+'</div>'+(job.state==='failed'?'<div class="failure"><h2>本次计算未完成</h2><p>'+esc(job.error)+'</p><p>已算班次保留，重试会从断点续跑，不会重复计入。</p></div>':job.state!=='completed'?'<div class="pending"><span class="spinner"></span><h2>独立 worker 正在分批计算</h2><p>页面可稍后刷新；中断后任务会按已算班次继续。</p></div>':segmentTable(job.result))+'</section>':'');
}

export function bindDrift(refresh){
  bind('#drift-form','submit',async event=>{event.preventDefault();const form=event.target;const button=form.querySelector('button[type="submit"]');if(button)button.disabled=true;try{const data=Object.fromEntries(new FormData(form));const job=await driftApi.create({run_id:Number(data.run_id),label:data.label,penalty:Number(data.penalty)});research.driftJobId=job.id;await refresh();toast('漂移任务已进入队列')}catch(e){toast(e.message,true)}finally{if(button)button.disabled=false}});
  action('#plan-form',async form=>{const data=Object.fromEntries(new FormData(form));const shifts=parseAssignments(data.assignments);await driftApi.createPlan({source_id:Number(data.source_id),note:data.note,shifts});await refresh();toast('归班新版本已保存')});
  for(const a of document.querySelectorAll('[data-drift-job]'))a.onclick=async()=>{research.driftJobId=Number(a.dataset.driftJob);await refresh()};
  bind('#drift-retry','click',async()=>{try{await driftApi.retry(research.driftJobId);await refresh();toast('已按断点续跑')}catch(e){toast(e.message,true)}});
  bind('#drift-publish','click',async()=>{try{await driftApi.publish(research.driftJobId);await refresh();toast('已按版本发布最新分界')}catch(e){toast(e.message,true)}});
}
