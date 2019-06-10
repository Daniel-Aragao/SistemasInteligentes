import random


class Crossover:
    
    @staticmethod
    def OBX(father, mother, selected_elements=None):
        # father = ['A', 'B', 'C', 'D', 'F', 'E', 'G']
        # mother = ['C', 'E', 'G', 'A', 'D', 'F', 'B']
        # selected_elements = [4 , 1, 3]
        # return (['A', 'D', 'C', 'F', 'B', 'E', 'G'], ['C', 'A', 'G', 'D', 'E', 'F', 'B'])
        
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
            
        return son, daughter
        
    @staticmethod
    def OX(father, mother, selected_elements=None):
        # father = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # mother = [9, 3, 7, 8, 2, 6, 5, 1, 4]
        # selected_elements = [3, 6]
        # return tuple([[3, 8, 2, 4, 5, 6, 7, 1, 9]])
        
        son = [i for i in father]
        father_out_of_cut = []
        mother_order_from_right_cut = []
        
        chromossome_size = len(son)
        
        if not selected_elements:
            cut_left = random.randint(0, chromossome_size - 1)
            cut_right = -1
            
            while cut_right < cut_left:
                cut_right = random.randint(cut_left, chromossome_size - 1)
        else:
            cut_left, cut_right = selected_elements
          
        for i in range(0, chromossome_size):
            if i < cut_left or i > cut_right:
                father_out_of_cut.append(father[i])
            
            if i <= cut_right:
                mother_order_from_right_cut.append(mother[i])
            elif i > cut_right:
                mother_order_from_right_cut.insert(0, mother[i])
                
        ordered_elements = sorted(father_out_of_cut, key=lambda e: mother_order_from_right_cut.index(e))
        
        ordered_elements.reverse()
            
        for i in range(cut_right + 1, chromossome_size):
            son[i] = ordered_elements.pop()
            
        for i in range(0, cut_left):
            son[i] = ordered_elements.pop()
            
        return tuple([son])