import glob
files = glob.glob("*.hgu1")
for file in files:
    HgulFile = open(file, "rb")
    FileHeader = HgulFile.read(8)
    Code = HgulFile.read(2)
    name = 1
    while Code != b'':
        Output = open("_"+file[:-5]+"_"+str(name)+".csv", "w")
        Width = HgulFile.read(1)
        Height = HgulFile.read(1)
        reserved = HgulFile.read(2)
        for i in range(ord(Height)):
            for j in range(ord(Width)):
                temp = HgulFile.read(1)
                if j != ord(Width) - 1:
                    Output.write("%3d,"%ord(temp))
                else:
                    Output.write("%3d"%ord(temp))
            Output.write("\n")
        Code = HgulFile.read(2)
        name += 1
        Output.close()
    HgulFile.close()
    print("finish " + file)
