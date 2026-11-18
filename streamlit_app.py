"""Dark, self-contained Streamlit dashboard for humanitarian RL routing."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.inference import ACTION_DISPLAY_NAMES, ACTION_NAMES, AGENT_METADATA, ModelRegistry, load_training_metrics, run_showdown, run_single_simulation
from viewer_3d import render_route_viewer

st.set_page_config(page_title="Humanitarian RL Command Center", page_icon="✈️", layout="wide")
st.markdown("""<style>.stApp{background:#070b14;color:#dbeafe}.block-container{padding-top:1.4rem}.stMetric{background:#0d1729;border:1px solid #203555;border-radius:12px;padding:14px}.status{padding:6px 10px;border-radius:999px;font-weight:700}</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading RL policy registry…")
def registry(): return ModelRegistry.get_instance()

def obj(x): return x.model_dump() if hasattr(x,"model_dump") else x.dict()

st.title("Humanitarian RL Command Center")
st.caption("Risk-aware dispatch simulation · Dubai Origin → Regional Checkpoint → Disaster Site")
with st.sidebar:
    st.header("What-If Dispatch Controls")
    tier_label=st.selectbox("Urgency Tier",["Tier 1: Critical Emergency","Tier 2: Standard"])
    weight=st.slider("Shipment Weight (tons)",.5,50.0,15.0,.5)
    volume=st.slider("Shipment Volume",1.0,100.0,30.0,1.0)
    deadline=st.slider("Deadline Remaining (days)",1.0,30.0,12.0,.5)
    shock=st.slider("Corridor Disruption Shock Probability",0.0,.5,.35,.01)
    seed=st.number_input("Random Seed",0,999999,42,1)
    if st.button("Roll New Seed",use_container_width=True): seed=random.randint(0,999999)
    agent_id=st.selectbox("Active Agent",list(AGENT_METADATA),format_func=lambda x:AGENT_METADATA[x]["name"])
    st.caption("Missing checkpoints load initialized policies; a warning is shown below.")

scenario=dict(urgency_tier=1.0 if tier_label.startswith("Tier 1") else 2.0,shipment_weight=weight,deadline_remaining=deadline,disruption_shock_prob=shock,seed=int(seed),volume=volume)
try:
    sim=run_single_simulation(agent_id,**scenario)
except Exception as exc:
    st.error(f"Unable to run inference: {exc}"); st.stop()

tabs=st.tabs(["Live 3D Route Simulator","Quantile & CVaR Risk Inspector","7-Model Showdown","Training Diagnostics"])
with tabs[0]:
    a,b,c,d=st.columns(4); a.metric("Total Freight Cost",f"${sim.total_freight_cost:,.0f}"); b.metric("Elapsed Days",f"{sim.total_days:.1f} / {sim.initial_deadline:.1f}")
    violated=sim.status=="Deadline Violated"; c.markdown(f"<div class='status' style='background:{'#4c1018' if violated else '#064e3b'}'>{'VIOLATED' if violated else 'ON TIME'}</div>",unsafe_allow_html=True); d.metric("Cargo Weight",f"{weight:.1f} t")
    render_route_viewer(sim.trajectory)
    rows=[obj(x) for x in sim.trajectory]
    st.dataframe(pd.DataFrame([{"Action":r["action_name"],"Transit Days":round(r["days_passed"],2),"Step Cost":f"${r['freight_cost']:,.0f}","Remaining Slack":round(r["deadline_remaining"],2)} for r in rows]),use_container_width=True,hide_index=True)
with tabs[1]:
    alpha=st.radio("Tail-risk α",[.05,.10],format_func=lambda x:f"{int(x*100)}%",horizontal=True)
    state=sim.initial_state; q=registry().extract_quantiles_analysis(state); qd=obj(q); cutoff=qd["cutoff_index_05"] if alpha==.05 else qd["cutoff_index_10"]; scores=qd["cvar_05"] if alpha==.05 else qd["cvar_10"]
    fig=go.Figure();
    for i,values in enumerate(qd["quantiles"]): fig.add_trace(go.Scatter(x=list(range(1,51)),y=sorted(values),mode="lines+markers",name=ACTION_NAMES[i]))
    fig.add_vrect(x0=.5,x1=cutoff+.5,fillcolor="#ef4444",opacity=.14,line_width=0,annotation_text=f"Worst {int(alpha*100)}% tail")
    fig.update_layout(template="plotly_dark",height=470,xaxis_title="Predicted return quantile (Z₁ … Z₅₀)",yaxis_title="Return")
    st.plotly_chart(fig,use_container_width=True)
    mask=sim.trajectory[0].action_mask
    st.dataframe(pd.DataFrame({"Action":ACTION_DISPLAY_NAMES,"Mean Q-Value":qd["q_mean"],f"CVaR {int(alpha*100)}% Score":scores,"Legal Action Mask":["Legal" if x else "Blocked" for x in mask]}).round(3),use_container_width=True,hide_index=True)
with tabs[2]:
    if st.button("Run all 7 policies on this exact scenario",type="primary") or "showdown" not in st.session_state:
        try: st.session_state.showdown=run_showdown(**scenario)
        except Exception as exc: st.error(f"Showdown failed: {exc}")
    if "showdown" in st.session_state:
        data=[obj(x) for x in st.session_state.showdown.results]; df=pd.DataFrame(data); x=df.agent_name
        left,right=st.columns(2); left.plotly_chart(go.Figure(go.Bar(x=x,y=df.total_freight_cost,marker_color="#22d3ee")).update_layout(template="plotly_dark",title="Freight Cost ($)"),use_container_width=True); right.plotly_chart(go.Figure(go.Bar(x=x,y=df.total_days,marker_color="#fbbf24")).add_hline(y=deadline,line_dash="dash",annotation_text="Deadline").update_layout(template="plotly_dark",title="Transit Days vs Deadline"),use_container_width=True)
        st.dataframe(df[["agent_name","status","total_freight_cost","total_days","deadline_remaining","pivoted"]],use_container_width=True,hide_index=True)
with tabs[3]:
    csv=Path("results/logs/training_metrics.csv")
    if not csv.exists(): st.warning("Training diagnostics file is not available.")
    else:
        df=pd.read_csv(csv); metrics=[("Episode Reward","reward"),("Quantile Huber Loss","loss"),("Tier-1 Violation Rate","violation"),("Lagrangian Multiplier (λ)","lambda_penalty"),("Exploration (ε)","epsilon")]
        for title,col in metrics:
            y=df[col].rolling(100,min_periods=1).mean()*100 if col=="violation" else df[col]
            st.plotly_chart(go.Figure(go.Scatter(x=df.episode,y=y,mode="lines",line=dict(color="#38bdf8"))).update_layout(template="plotly_dark",title=title,height=250,xaxis_title="Episode"),use_container_width=True)
