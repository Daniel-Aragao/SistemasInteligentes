import random


class Mutation:
    
    @staticmethod
    def position_based(son, selected_elements=None):
        # son = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        # selected_elements = [1, 4]
        # son = ['A', 'C', 'D', 'E', 'B', 'F', 'G']
        
        chromossome_size = len(son)
        
        if not selected_elements:
            i = random.randint(0, chromossome_size - 1)
            j = i
            
            while i == j:
                j = random.randint(0, chromossome_size - 1)
        else:
            i,j = selected_elements
        
        aux = son[i]
        son.remove(aux)
        son.insert(j, aux)
    
    @staticmethod
    def inversion(son, selected_elements=None):
        # son = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        # selected_elements = [2, 4]
        # son = ['A', 'B', 'E', 'D', 'C', 'F', 'G']
        chromossome_size = len(son)
        
        if not selected_elements:
            cut_left = random.randint(0, chromossome_size - 1)
            cut_right = -1
            
            while cut_right < cut_left:
                cut_right = random.randint(cut_left, chromossome_size - 1)
        else:
            cut_left, cut_right = selected_elements
            
        sublist = son[cut_left : cut_right + 1:][::-1]
        
        for i in range(cut_left, cut_right + 1):
            son[i] = sublist[i - cut_left]
        