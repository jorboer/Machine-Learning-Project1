import matplotlib.pyplot as plt
import numpy as np
import helpers


yb, input_data, ids = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv")
#yb_test, input_data_test, ids_test = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\test.csv")

higgs_bosons = []
ids_higgs = []
remainder = []
ids_remainder = []
for i in range(len(input_data)):
    if(yb[i] == 1):
        higgs_bosons.append(input_data[i])
        ids_higgs.append(ids[i])
    else:
        remainder.append(input_data[i])
        ids_remainder.append(ids[i])

standard_deviations_higgs = np.std(higgs_bosons, axis=0)
standard_deviations_remainder = np.std(remainder, axis=0)
mean_higgs = np.mean(higgs_bosons, axis=0)
mean_remainder = np.mean(remainder, axis=0)
variance_higgs = np.var(higgs_bosons, axis=0)
variance_remainder = np.var(remainder, axis=0)

teller=1
print("\n\n\n")
print("      STANDARD DEVIATION")
print("\n")
print("   HIGGS(s)      REMAINDER(b)")


for i in range(len(standard_deviations_higgs)):
    print(teller, " ", standard_deviations_higgs[i], " ", standard_deviations_remainder[i])
    teller += 1

print("\n\n\n")
print("           MEAN")
print("\n")
print("   HIGGS(s)      REMAINDER(b)")


teller = 1
for i in range(len(mean_higgs)):
    print(teller, " ", mean_higgs[i], " ", mean_remainder[i])
    teller += 1

print("\n\n\n")
print("         VARIANCE")
print("\n")
print("   HIGGS(s)      REMAINDER(b)")

teller = 1
for i in range(len(variance_higgs)):
    print(teller, " ", variance_higgs[i], " ", variance_remainder[i])
    teller += 1
"""
correlation = []
remainder = remainder[:len(higgs_bosons)]

higgs_bosons_transpose = np.transpose(higgs_bosons)
remainder_transpose = np.transpose(remainder)

print("\n\n\n")

for i in range(len(higgs_bosons_transpose)):
    correlation.append(np.corrcoef(higgs_bosons_transpose[i],remainder_transpose[i]))
    print(np.corrcoef(higgs_bosons_transpose[i],remainder_transpose[i]))"""