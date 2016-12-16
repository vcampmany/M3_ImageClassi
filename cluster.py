import os, glob

if __name__ == 'main':
    #Cluster parameters
    queue_name = 'fast.master.q'
    memory = '5M'


    current_cmd = 'session_cluster.py 100 30 20 linear 1 None sift'
    cmd = 'qsub -cwd -V -q %s -l mem_token=%s,mem_free=%s %s' % ( queue_name, memory, memory, current_cmd)

    print os.popen(cmd).read()