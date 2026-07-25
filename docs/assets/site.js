"use strict";
const profiles={
  demo:[["Input","6 × 3 × 8 × 8","procedural camera planes"],["World memory","8 × 8 × 16","temporal BEV"],["Queries","64 → top 8","stable tracking selection"],["Future","3 × 4","motion modes × steps"],["Action","6 points","plan + collision score"]],
  production:[["Input","6 × BGR","normalized + padded views"],["World memory","200 × 200 × 256","temporal BEV"],["Queries","900","learned tracking queries"],["Future","6 × 12","motion modes × steps"],["Action","6 points","released planning horizon"]]
};
document.querySelectorAll("[data-profile]").forEach(button=>button.addEventListener("click",()=>{
  document.querySelectorAll("[data-profile]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));
  document.querySelector("#pipeline").innerHTML=profiles[button.dataset.profile].map(x=>
    `<li><small>${x[0]}</small><strong>${x[1]}</strong><span>${x[2]}</span></li>`).join("");
}));
let inventory=[];
const body=document.querySelector("#operators");
function renderInventory(query=""){body.innerHTML=inventory.filter(row=>Object.values(row).join(" ").toLowerCase().includes(query.toLowerCase())).map(row=>
  `<tr><td>${row.operator}</td><td>${row.production}</td><td>${row.demo}</td><td>${row.route}</td></tr>`).join("")}
fetch("assets/operator-inventory.json").then(r=>r.json()).then(data=>{inventory=data;renderInventory()}).catch(()=>{});
document.querySelector("#filter").addEventListener("input",event=>renderInventory(event.target.value));
const canvas=document.querySelector("#bev"),ctx=canvas.getContext("2d"),reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;
function draw(result,t=1){
  const w=canvas.width,h=canvas.height,scale=48,ox=w/2,oy=h-55;
  ctx.clearRect(0,0,w,h);ctx.strokeStyle="#283029";ctx.lineWidth=1;
  for(let x=ox-8*scale;x<=ox+8*scale;x+=scale){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
  for(let y=oy-10*scale;y<=oy+2*scale;y+=scale){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
  const xy=p=>[ox+p[0]*scale,oy-p[1]*scale];
  result.map.forEach(m=>{ctx.strokeStyle="#626c62";ctx.beginPath();m.points.forEach((p,i)=>{const q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke()});
  result.motion.forEach(m=>m.modes.forEach((mode,j)=>{ctx.strokeStyle=`rgba(132,148,118,${.7-j*.18})`;ctx.beginPath();mode.trajectory.forEach((p,i)=>{const q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke()}));
  ctx.strokeStyle="#c9ff48";ctx.lineWidth=4;ctx.beginPath();result.ego_plan.forEach((p,i)=>{const q=xy(p);i?ctx.lineTo(q[0],oy+(q[1]-oy)*t):ctx.moveTo(q[0],oy+(q[1]-oy)*t)});ctx.stroke();
  result.tracks.forEach(track=>{const q=xy([track.x,track.y]);ctx.fillStyle="#ff6a3d";ctx.beginPath();ctx.arc(q[0],q[1],5,0,Math.PI*2);ctx.fill()});
  ctx.fillStyle="#c9ff48";ctx.fillRect(ox-7,oy-16,14,32)
}
fetch("assets/demo-result.json").then(r=>r.json()).then(result=>{
  document.querySelector("#viz-status").textContent=`frame ${result.frame_index} · ${result.tracks.length} tracks · collision ${result.collision_score.toFixed(3)}`;
  if(reduce){draw(result)}else{let start=performance.now();function tick(now){draw(result,.6+.4*Math.min(1,(now-start)/1200));if(now-start<1200)requestAnimationFrame(tick)}requestAnimationFrame(tick)}
}).catch(()=>document.querySelector("#viz-status").textContent="Canonical result unavailable.");
