import random


class Crossover:
    
    @staticmethod
    def OBX(father, mother, selected_elements=None):
        # father = ['A', 'B', 'C', 'D', 'F', 'E', 'G']
        # mother = ['C', 'E', 'G', 'A', 'D', 'F', 'B']
        son = [i for i in father]
        daughter = [i for i in mother]
        
        if not selected_elements:
            chromossome_size = len(son)
            crossover_elements = random.randint(0, chromossome_size)
            
            selected_elements = []
            
            while len(selected_elements) < crossover_elements:
                selected_element = random.randint(0, chromossome_size - 1)
                
                if not selected_element in selected_elements:
                    selected_elements.append(selected_element)
        
        selected_elements_son = [son[i] for i in selected_elements]
        selected_elements_daughter = [daughter[i] for i in selected_elements]
        
        selected_elements_son = sorted(selected_elements_son, key=lambda e: daughter.index(e))
        selected_elements_daughter = sorted(selected_elements_daughter, key=lambda e: son.index(e))
        
        for i, e in enumerate(sorted(selected_elements)):
            son[e] = selected_elements_son[i]
            daughter[e] = selected_elements_daughter[i]
            
        print(selected_elements)
            
        return son, daughter
        