import csv
import urllib
import os


#must be csv
file = 'observations-58064.csv'


def download_pics():
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i in range(60):
                print(csv_reader.next())
            line_count = 0
            name = 1

            for row in csv_reader:
                print(row)
                print(line_count)
                if line_count == 0:
                    line_count +=1
                else:
                    url = row[13]
                    sci_name = row[36]

                    print('downloading pic: '+ str(line_count) + "  from: "+ str(url))
                    
                    if not os.path.exists("./photos/raw/%s" % sci_name): 
                        os.makedirs("./photos/raw/%s" % sci_name)

                    urllib.urlretrieve(url,'./photos/raw/%s/%s.jpg' % (sci_name, name))
                    line_count += 1
                name +=1




download_pics()




