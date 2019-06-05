class Importer:
    
    def import_cities(path):
        with open(path) as file_obj:
            cities = []
            cities_size = 0
            
            for line in file_obj:
                clean_line = line.strip()
                
                if clean_line:
                    splited_line = clean_line.split()
                    
                    if(not cities_size):
                        cities_size = int(splited_line[0])
                    else:
                        cities.append((splited_line[0], splited_line[1]))
                        
            
        return cities
