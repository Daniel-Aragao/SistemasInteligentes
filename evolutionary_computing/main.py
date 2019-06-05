from importer import Importer

path = "misc/ncit30.dat"

def main():
    N = 50 # tamanho da população inicial
    generations_limit = 200 # limite de gerações
    pc = 0.75 # probabilidade de crossover
    pm = 0.1 # probabilidade de mutação
    
    cities = Importer.import_cities(path)
    
    
    random_seed = 1
    
    


if __name__ == "__main__":
    main()