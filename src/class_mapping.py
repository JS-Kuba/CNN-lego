class ClassMapping():
    mapping = {
        0: "flower",
        1: "leaf",
        2: "stone",
        3: "wood"
    }
    @classmethod
    def get_label(cls, class_idx):
        if class_idx not in range(4):
            print("Wrong class index - cannot map to label!")
            return None
        else:
            return cls.mapping.get(class_idx)

    