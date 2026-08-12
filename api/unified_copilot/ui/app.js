const $=s=>document.querySelector(s), app=$('#app'), turns=$('#turns'), input=$('#textInput');
let lastTurns='', mode='DORMANT';
// 서버는 최근 12턴만 보관한다. 길이만 비교하면 12턴을 채운 뒤로는 값이 영원히
// 12라서, 새 답변이 와도 다시 그리지 않는다 - 대화가 화면에서 멈춰 보이던 원인.
// 마지막 턴의 시각까지 함께 비교하면 내용이 밀려도 매번 갱신된다.
const turnKey=c=>c.length+':'+(c.length?c[c.length-1].at:0);
function particles(){const c=$('#field'),x=c.getContext('2d');let ps=[];function size(){c.width=innerWidth;c.height=innerHeight;ps=Array.from({length:90},()=>({x:Math.random()*c.width,y:Math.random()*c.height,z:Math.random(),v:.08+Math.random()*.25}))}function draw(){x.clearRect(0,0,c.width,c.height);for(const p of ps){p.y-=p.v*(1+p.z*2);if(p.y<0)p.y=c.height;x.fillStyle=`rgba(91,226,255,${.06+p.z*.25})`;x.fillRect(p.x,p.y,1+p.z,1+p.z)}requestAnimationFrame(draw)}addEventListener('resize',size);size();draw()}particles();
setInterval(()=>$('#clock').textContent=new Date().toLocaleTimeString('ko-KR',{hour12:false}),1000);
function render(s){const a=s.assistant,w=s.assembly;mode=a.mode;app.className=(w.active?'work ':'')+a.mode.toLowerCase();$('#modeLabel').textContent=`NEURAL INTERFACE · ${a.mode}`;$('#heroTitle').textContent=a.mode==='DORMANT'?'대기 모드':a.mode==='THINKING'?'분석 중':a.mode==='ERROR'?'연결 확인':'JARVIS 온라인';$('#status').textContent=a.message;$('#apiState').textContent=a.api_available?'API ONLINE':'API OFFLINE · LOCAL TEST';input.placeholder=a.mode==='DORMANT'?'JARVIS라고 입력해 깨우기':'무엇이든 입력하세요';$('#product').textContent=w.product||'작업 코파일럿';$('#stepNo').textContent=w.step??'—';$('#stepTotal').textContent=w.total?`/ ${w.total} STEP`:'STEP';$('#stepTitle').textContent=w.title||'작업 대기';$('#instruction').textContent=w.instruction||'작업을 시작하면 현재 단계가 표시됩니다.';renderHeard(s.heard);renderReference(s.reference);if(turnKey(s.conversation)!==lastTurns){lastTurns=turnKey(s.conversation);turns.innerHTML=s.conversation.map(t=>`<div class="turn ${t.role}"><b>${t.role==='user'?'USER':'JARVIS'}</b>${escapeHtml(t.text)}</div>`).join('');turns.scrollTop=turns.scrollHeight}}
function escapeHtml(v){const d=document.createElement('div');d.textContent=v;return d.innerHTML}
// 마이크가 무엇을 들었는지 항상 보여준다. 아무 것도 안 뜨면 소리가 안 들어오는
// 것이고, 노란 줄이 뜨면 들리기는 했는데 호출어가 없어 무시된 것이다.
function renderHeard(h){const el=$('#heard');if(!h||!h.text){el.textContent='';el.className='heard';return}
const age=(Date.now()/1000)-(h.at||0);if(age>25){el.textContent='';el.className='heard';return}
el.className='heard '+(h.gated?'gated':'live');
el.innerHTML='<b>'+(h.gated?'들림 · 호출어 없어 무시':'들림')+'</b>'+escapeHtml(h.text)}
// 참고 사진 패널. version이 바뀔 때만 이미지를 다시 받는다 - 매 폴링마다
// 새로 받으면 250ms 간격으로 사진을 다시 내려받게 된다.
let referenceVersion=-1;
function renderReference(r){const panel=$('#referencePanel');if(!r){panel.hidden=true;return}
if(r.version!==referenceVersion){referenceVersion=r.version;
if(r.available)$('#referenceImg').src='/api/reference.jpg?v='+r.version}
$('#referenceLabel').textContent=r.label||'';panel.hidden=!r.available&&!r.label}
async function poll(){try{const s=await fetch('/api/state',{cache:'no-store'}).then(r=>r.json());render(s)}catch{}setTimeout(poll,250)}poll();
setInterval(()=>{$('#camera').src='/api/frame.jpg?t='+Date.now()},350);
$('#inputForm').addEventListener('submit',async e=>{e.preventDefault();const text=input.value.trim();if(!text)return;input.value='';const r=await fetch('/api/input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});const out=await r.json();if(out.reason==='wake_word_required'){input.placeholder='먼저 JARVIS라고 불러 주세요';input.classList.add('reject');setTimeout(()=>input.classList.remove('reject'),700)}});
$('#sleepBtn').onclick=()=>fetch('/api/sleep',{method:'POST'});
// A restrained audio-reactive idle motion; real TTS PCM can drive this value later.
setInterval(()=>{const m=$('.mouth');if(mode==='SPEAKING'){m.style.height=`${5+Math.random()*13}%`;m.style.transform=`scaleX(${.8+Math.random()*.35})`}else{m.style.height='7%';m.style.transform=''}},90);
