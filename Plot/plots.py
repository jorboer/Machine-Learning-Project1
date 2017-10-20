import helpers
import matplotlib.pyplot as plt

data_path = "C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv"

#load training data
yb, input_data, ids = helpers.load_csv_data(data_path)

#remove unmeasured values
#input_data = processing.preprocess_training_data(input_data, "mean")

#standardize inputdata
stand_data, mean_data, std_data = helpers.standardize(input_data)

#print(np.mean(stand_data, axis=0))

#blue = signal, red = background
c1 = ['blue' if x == 1 else 'red' for x in yb]



#for x1 in range(input_data.shape[1]):
for x1 in range(2,input_data.shape[1]):
    for x2 in range(input_data.shape[1]):
        plt.scatter(stand_data[:,x1], stand_data[:,x2], color=c1, s=0.2)
        plt.xlabel(x1+1)
        plt.ylabel(x2+1)
        plt.savefig("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\plots1\\"
                    + str(x1+1) + "-" + str(x2+1) + " plot.png", dpi=200)
        plt.clf()

"""yb, input_data, ids = helpers.load_csv_data("train.csv")
#yb_test, input_data_test, ids_test = helpers.load_csv_data("test.csv")

higgs_bosons = np.array()
ids_higgs = np.array()
remainder = []
ids_remainder = []
for i in range(len(input_data)):
    if(yb[i] == 1):
        higgs_bosons.append(input_data[i])
        ids_higgs.append(ids[i])
    else:
        remainder.append(input_data[i])
        ids_remainder.append(ids[i])

plt.scatter(ids_higgs, higgs_bosons[:,0])
plt.show()"""