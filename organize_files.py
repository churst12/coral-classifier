import os
import shutil
src_files = os.listdir('./photos/testing')
print(src_files)
for dir_name in src_files:
	print(dir_name)
	if dir_name ==  '.DS_Store':
		pass
	else:

		genus_path = './photos/testing/%s' % dir_name
		genus = os.listdir(genus_path)
		
		for species in genus:
			if species ==  '.DS_Store':
				pass
			else:
				species_path = '%s/%s' % (genus_path, species)
				species = os.listdir(species_path)

				for photo in species:
					print(photo)


					full_file_name = os.path.join(species_path, photo)
					print(full_file_name)
					if os.path.isfile(full_file_name):
						shutil.copy(full_file_name, genus_path)