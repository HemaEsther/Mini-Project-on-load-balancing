import pandas as pd

before = pd.read_csv('edge_dataset.csv')
after = pd.read_csv('edge_dataset_after_assignments.csv')
log = pd.read_csv('assignment_log.csv')

# aggregate per node
summary = []
for df, label in [(before, 'before'), (after, 'after')]:
    g = df.groupby('node')['load'].agg(['mean','std','max','min']).reset_index()
    g['label'] = label
    summary.append(g)

summary_df = pd.concat(summary)
print('Per-node load summary (before vs after):')
print(summary_df.sort_values(['node','label']).to_string(index=False))

# overall stats
print('\nOverall load stats:')
print('Before mean:', before['load'].mean(), 'After mean:', after['load'].mean())
print('Before std :', before['load'].std(), 'After std :', after['load'].std())

# assignment counts
assign_counts = log['selected_node'].value_counts().sort_index()
print('\nAssignments per node:')
print(assign_counts.to_string())

# average increase of selected nodes at assignment times
# compare new_load - previous load at the same time and node in before dataset
inc_list = []
for idx, r in log.iterrows():
    t = r['time']
    n = r['selected_node']
    job_load = r['job_load']
    new_load = r['new_load']
    prev = before[(before['time']==t)&(before['node']==n)]
    if not prev.empty:
        prev_load = prev.iloc[0]['load']
        inc_list.append({'time':t,'node':n,'prev_load':prev_load,'job_load':job_load,'new_load':new_load,'delta':new_load-prev_load})

inc_df = pd.DataFrame(inc_list)
print('\nSample of assignments and deltas:')
print(inc_df.head().to_string(index=False))
print('\nAverage delta after assignment:', inc_df['delta'].mean())

# simple improvement metric: std across nodes at each time step before vs after
stds_before = before.groupby('time')['load'].std().mean()
stds_after = after.groupby('time')['load'].std().mean()
print('\nMean per-timestep std across nodes: before=', stds_before, ' after=', stds_after)

# save a small CSV summary
summary_df.to_csv('improvement_summary.csv', index=False)
print('\nWrote improvement_summary.csv')
