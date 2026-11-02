"""Read-only view of analyze.py exports. Run with streamlit run."""
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
ANALYSIS=Path(os.environ.get('RF_ANALYSIS_DIR', str(ROOT/'analysis')))
SNAPSHOT=Path(os.environ.get('RF_SNAPSHOT', str(ROOT/'data/inputs.json')))
st.set_page_config(page_title='IndicJudge | Reference-free evaluation',layout='wide')
st.title('Reference-free evaluation')
st.caption('Candidate performance and judge quality are different comparisons. These tables show the latest saved analysis, not live model calls.')
if st.button('Refresh saved analysis'):st.rerun()
report_path=ANALYSIS/'coverage.json'
if not report_path.exists():
    st.info('Prepare and run the judges, then run python3 run.py analyze from the project root.');st.stop()
report=json.loads(report_path.read_text())
snapshot=SNAPSHOT
if snapshot.exists() and json.loads(snapshot.read_text())['snapshot_hash']!=report['snapshot_hash']:
    st.error('Analysis belongs to a different snapshot. Re-run analyze.py.');st.stop()
cols=st.columns(3)
cols[0].metric('Frozen responses',report['candidate_count'])
cols[1].metric('Rated by all three judges',report['matched_for_all_judges'])
cols[2].metric('Adjudicated human ratings',report['human_adjudicated'])
st.write('Saved judgments:',report['judge_counts'])
quality=pd.read_csv(ANALYSIS/'judge_leaderboard.csv')
agreement=pd.read_csv(ANALYSIS/'judge_agreement.csv')
candidates=pd.read_csv(ANALYSIS/'candidate_scores.csv')
lang=st.selectbox('Language',['all']+sorted(x for x in agreement.language_code.unique() if x!='all'))
task=st.selectbox('Task',['all']+sorted(x for x in agreement.task.unique() if x!='all'))
def selected(frame):return frame[(frame.language_code==lang)&(frame.task==task)]
tabs=st.tabs(['Judge quality against humans','Judge agreement','Candidate scores'])
with tabs[0]:
    st.write('Primary ordering: lower MAE against adjudicated human scores. Quadratic weighted kappa and within-one-point agreement provide supporting evidence.')
    rows=selected(quality)
    if rows.empty:st.info('No matched human ratings for this group. A best-judge ranking cannot be established yet.')
    else:
        st.dataframe(rows.sort_values('mae'),hide_index=True,use_container_width=True)
        st.caption('Ranks are descriptive. Pointwise confidence intervals do not establish a winner across multiple comparisons. A blank rank means too few sample clusters.')
        delta=pd.read_csv(ANALYSIS/'paired_mae_differences.csv')
        st.dataframe(selected(delta),hide_index=True,use_container_width=True)
with tabs[1]:
    st.dataframe(selected(agreement),hide_index=True,use_container_width=True)
    st.caption('Agreement is not accuracy. All pairs use the same rows rated by all three judges. Blank kappa means it was undefined.')
with tabs[2]:
    st.dataframe(selected(candidates),hide_index=True,use_container_width=True)
    st.caption('Higher candidate scores reflect a judge’s assessment, not that judge’s accuracy. Both Gemmas use the same samples, rated by all three judges.')
st.caption(report['weighting'])
st.caption(report['limitations'])
