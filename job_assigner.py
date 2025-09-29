def assign_job_to_node(df, node, time_step, job_load):
    mask = (df["node"]==node) & (df["time"]==time_step)
    if mask.sum() == 0:
        return df
    df.loc[mask, "load"] += job_load
    df.loc[mask, "active_sessions"] += int(round(job_load/3))
    df.loc[mask, "cpu"] = (df.loc[mask, "cpu"] + job_load*0.6).clip(0,100)
    df.loc[mask, "mem"] = (df.loc[mask, "mem"] + job_load*0.4).clip(0,100)
    return df