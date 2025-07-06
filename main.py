from incar_agent import generate_incar

if __name__ == "__main__":
    task = "electronic_structure"
    material = "Cu(111)"

    print(f"Generating INCAR for material: {material}, task: {task}...\n")
    incar = generate_incar(task, material)

    print("===== INCAR OUTPUT =====")
    print(incar)