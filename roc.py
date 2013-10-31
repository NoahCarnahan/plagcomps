import matplotlib
matplotlib.use('pdf')
#['ps', 'Qt4Agg', 'GTK', 'GTKAgg', 'svg', 'agg', 'cairo', 'MacOSX', 'GTKCairo', 'WXAgg', 'TkAgg', 'QtAgg', 'FltkAgg', 'pdf', 'CocoaAgg', 'emf', 'gdk', 'template', 'WX']
import matplotlib.pyplot as pyplot

#http://scikit-learn.org/0.11/auto_examples/plot_roc.html
#http://stackoverflow.com/questions/8192455/ntlk-python-plotting-roc-curve
#http://gim.unmc.edu/dxtests/roc2.htm

#fpr = fp/(fp+tn)    # false positive rate
#tpr = tp/(tp+fn)    # true positive rate



#We want a point for each threshold:
# - avg fpr
#   calculate by averaging the fpr for each file
# - avg tpr
 

fpr_avgs = []
tpr_avgs = []
thresholds = ["0", "0.25", "0.4", "0.5", "0.6", "0.75"]

for t in thresholds:
    total = 0
    fpr_sum = 0.0
    tpr_sum = 0.0
    f = file("results/noah_test_"+t+".csv")
    for line in f:
        parts = line.split(",")
        if parts[-1] != " false_neg\n":
            fn = int(parts[-1])
            tn = int(parts[-2])
            fp = int(parts[-3])
            tp = int(parts[-4])
            try:
                fpr = fp/float(fp+tn)
            except ZeroDivisionError:
                fpr = 0
            try:
                tpr = tp/float(tp+fn)
            except ZeroDivisionError:
                tpr = 0
            fpr_sum += fpr
            tpr_sum += tpr
            total += 1
    f.close()
    fpr_avgs.append(fpr_sum/total)
    tpr_avgs.append(tpr_sum/total)

print "fpr: ", fpr_avgs
print "tpr: ", tpr_avgs
    



pyplot.clf()
pyplot.plot(fpr_avgs, tpr_avgs, marker='o', color='r', ls='')
pyplot.plot([0, 1], [0, 1], 'k--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.0])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Receiver operating characteristic')
#pyplot.show()
pyplot.savefig("results/roc.pdf")



        
    