"use strict";

const $=(selector,root=document)=>root.querySelector(selector);
const $$=(selector,root=document)=>[...root.querySelectorAll(selector)];
const safe=text=>String(text).replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));

addEventListener("scroll",()=>{
  const max=document.documentElement.scrollHeight-innerHeight;
  $("#progress").style.width=`${max?scrollY/max*100:0}%`;
},{passive:true});

const stages=[
  {
    shape:"6 × image pyramid",
    copy:"六路 BGR 图像经 ResNet-101 与 FPN 形成多尺度相机特征；标定元数据决定每个 BEV query 应该从哪里采样。",
    boxes:[["6 cameras","surround view"],["R101 + FPN","multi-scale"],["Fcam","perspective"]]
  },
  {
    shape:"200 × 200 × 256",
    copy:"BEV queries 通过空间交叉注意力读取相机特征，并通过时序自注意力融合已按 ego motion 对齐的上一帧状态。",
    boxes:[["Fcam","multi-view"],["BEV Encoder","spatial + temporal"],["Bₜ","world state"]]
  },
  {
    shape:"900 object Q · 300 map Q",
    copy:"TrackFormer 维持动态 agent 身份，MapFormer 提取在线地图。两者不是终点：agent 与 map query 会继续进入运动预测。",
    boxes:[["Bₜ","shared state"],["TrackFormer","agent Q"],["MapFormer","map Q"]]
  },
  {
    shape:"6 × 12 trajectory · 5 occupancy",
    copy:"MotionFormer 建模 agent–agent、agent–map 与 agent–goal 交互；OccFormer 再把对象级未来投回稠密风险场。",
    boxes:[["Qagent + Qmap","sparse"],["MotionFormer","6 modes"],["OccFormer","dense future"]]
  },
  {
    shape:"6 × (x, y)",
    copy:"Planner 将 SDC query 与导航命令编码结合，预测六个 ego waypoint，并利用未来 occupancy 的碰撞代价约束最终路径。",
    boxes:[["SDC + command","intent"],["Planner","attention"],["τ*","6 waypoints"]]
  }
];
function renderStage(index){
  const stage=stages[index];
  $("#stage-canvas").innerHTML=`<div class="stage-diagram">${stage.boxes.map((box,i)=>
    `${i?'<span class="stage-arrow">→</span>':""}<div class="stage-box"><b>${safe(box[0])}</b><small>${safe(box[1])}</small></div>`).join("")}</div>`;
  $("#stage-shape").textContent=stage.shape;
  $("#stage-copy").textContent=stage.copy;
}
$$("[data-stage]").forEach((button,index)=>button.addEventListener("click",()=>{
  $$("[data-stage]").forEach(item=>item.setAttribute("aria-selected",String(item===button)));
  renderStage(index);
}));
renderStage(0);

const profiles={
  tiny:[
    ["Input","6 × 3 × 8 × 8","procedural camera planes"],
    ["World","8 × 8 × 16","scene-keyed temporal BEV"],
    ["Queries","64 → top 8","stable tracking selection"],
    ["Future","3 × 4","modes × prediction steps"],
    ["Action","6 points","plan + collision score"]
  ],
  production:[
    ["Input","6 × BGR","normalized + padded views"],
    ["World","200 × 200 × 256","temporal BEV"],
    ["Queries","900 + 300","object + map queries"],
    ["Future","6 × 12","motion · 5-step occupancy"],
    ["Action","6 points","occupancy-aware plan"]
  ]
};
function renderProfile(name){
  $("#tensor-pipe").innerHTML=profiles[name].map((item,index)=>
    `<div class="tensor-node"><small>0${index+1} · ${safe(item[0])}</small><b>${safe(item[1])}</b><span>${safe(item[2])}</span></div>`).join("");
}
$$("[data-profile]").forEach(button=>button.addEventListener("click",()=>{
  $$("[data-profile]").forEach(item=>item.setAttribute("aria-pressed",String(item===button)));
  renderProfile(button.dataset.profile);
}));
renderProfile("tiny");

$$(".occ").forEach(canvas=>{
  const ctx=canvas.getContext("2d"),step=Number(canvas.dataset.step),size=canvas.width;
  ctx.strokeStyle="#d3d0c7";ctx.lineWidth=1;
  for(let x=0;x<=size;x+=22){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,size);ctx.stroke()}
  for(let y=0;y<=size;y+=22){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(size,y);ctx.stroke()}
  const cells=[[5+step*.55,8-step*.7],[3+step*.25,4-step*.15],[7-step*.4,6+step*.35]];
  cells.forEach((cell,i)=>{
    const x=cell[0]*22,y=cell[1]*22;
    ctx.fillStyle=i===0?"rgba(228,75,55,.78)":"rgba(49,87,213,.45)";
    ctx.beginPath();ctx.ellipse(x,y,14+step*3,20+step*2,.2,0,Math.PI*2);ctx.fill();
  });
  ctx.fillStyle="#171b1c";ctx.fillRect(105,184,10,22);
});

let inventory=[];
const operatorBody=$("#operators");
function renderInventory(query=""){
  operatorBody.innerHTML=inventory.filter(row=>Object.values(row).join(" ").toLowerCase().includes(query.toLowerCase())).map(row=>
    `<tr><td>${safe(row.operator)}</td><td>${safe(row.production)}</td><td>${safe(row.demo)}</td><td>${safe(row.route)}</td></tr>`).join("");
}
fetch("assets/operator-inventory.json").then(response=>{
  if(!response.ok)throw new Error("inventory");
  return response.json();
}).then(data=>{inventory=data;renderInventory()}).catch(()=>{operatorBody.innerHTML='<tr><td colspan="4">Operator inventory unavailable.</td></tr>'});
$("#filter").addEventListener("input",event=>renderInventory(event.target.value));

const canvas=$("#bev"),ctx=canvas.getContext("2d");
function drawBev(result,progress=1){
  const w=canvas.width,h=canvas.height,scale=64,ox=w/2,oy=h-70;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#263033";ctx.lineWidth=1;
  for(let x=ox-9*scale;x<=ox+9*scale;x+=scale){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
  for(let y=oy-11*scale;y<=oy+2*scale;y+=scale){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
  const xy=point=>[ox+point[0]*scale,oy-point[1]*scale];
  (result.map||[]).forEach(item=>{
    ctx.strokeStyle="#718084";ctx.lineWidth=2;ctx.setLineDash([9,8]);ctx.beginPath();
    ctx.moveTo(...xy([item.x0,item.y0]));ctx.lineTo(...xy([item.x1,item.y1]));ctx.stroke();
  });
  ctx.setLineDash([]);
  (result.motion||[]).forEach(motion=>motion.modes.forEach((mode,index)=>{
    ctx.strokeStyle=`rgba(142,164,255,${.66-index*.16})`;ctx.lineWidth=2;ctx.beginPath();
    mode.trajectory.forEach((point,i)=>{const q=xy(point);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();
  }));
  ctx.strokeStyle="#f1db67";ctx.lineWidth=5;ctx.beginPath();
  (result.ego_plan||[]).forEach((point,index)=>{
    const q=xy(point),animated=[q[0],oy+(q[1]-oy)*progress];
    index?ctx.lineTo(...animated):ctx.moveTo(...animated);
  });ctx.stroke();
  (result.tracks||[]).forEach(track=>{
    const q=xy([track.x,track.y]);ctx.fillStyle="#e44b37";ctx.fillRect(q[0]-6,q[1]-9,12,18);
    ctx.fillStyle="#aebabc";ctx.font="11px monospace";ctx.fillText(`#${track.id}`,q[0]+10,q[1]+4);
  });
  ctx.fillStyle="#f4f2ec";ctx.fillRect(ox-8,oy-20,16,40);
}
fetch("assets/demo-result.json").then(response=>{
  if(!response.ok)throw new Error("result");
  return response.json();
}).then(result=>{
  $("#frame-value").textContent=result.frame_index;
  $("#track-value").textContent=result.tracks.length;
  $("#collision-value").textContent=Number(result.collision_score).toFixed(3);
  if(matchMedia("(prefers-reduced-motion: reduce)").matches){drawBev(result)}
  else{
    const start=performance.now();
    const tick=now=>{const p=Math.min(1,(now-start)/900);drawBev(result,.35+.65*p);if(p<1)requestAnimationFrame(tick)};
    requestAnimationFrame(tick);
  }
}).catch(()=>{$("#collision-value").textContent="n/a"});
