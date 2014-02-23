# 250
# 773


f = open("crisp_var_corp.txt", "r")
lines = f.readlines()
f.close()
suspicious_files = eval(lines[0])
source_files = eval(lines[1])
source_cutoff = eval(lines[2])

train_suspicious_files = suspicious_files[:250]
train_source_files = source_files[:773]
train_cutoff = source_cutoff[:250]

f = open("crisp_TRAIN_var_corp.txt", "w")
f.write(str(train_suspicious_files)+"\n"+str(train_source_files)+"\n"+str(train_cutoff))
f.close()
