import numpy as np

# 1. Definir el patrón ideal (Pelota de fútbol tradicional 4x4)
# -1 = Negro (Parches), 1 = Blanco (Cuero)
patron_ideal = np.array([
    [-1,  1,  1, -1],
    [ 1, -1, -1,  1],
    [ 1, -1, -1,  1],
    [-1,  1,  1, -1]
])

# Aplanar el patrón a un vector de 16 elementos
X_ideal = patron_ideal.flatten()
N = len(X_ideal)

# 2. Entrenar la Red de Hopfield (Calcular Matriz de Pesos W)
# Regla de Hebb: W = X * X.T
W = np.outer(X_ideal, X_ideal)
np.fill_diagonal(W, 0) # Eliminar auto-conexiones (W_ii = 0)

# 3. Definir el patrón ruidoso (Abstracción de la imagen distorsionada con "?" )
patron_ruidoso = np.array([
    [ 1,  1,  1, -1], # Esquina rota
    [ 1,  1, -1,  1], # Ruido en el centro por el signo "?"
    [ 1, -1, -1,  1],
    [-1,  1,  1,  1]  # Otra esquina con ruido
])
X_ruido = patron_ruidoso.flatten()

# 4. Proceso de Recuperación (Iteración de Hopfield)
X_actual = X_ruido.copy()
max_iteraciones = 5

print("--- ESTADO INICIAL (IMAGEN CON RUIDO) ---")
print(X_actual.reshape(4, 4))

for i in range(max_iteraciones):
    # Actualización sincrónica: Signo(W * X)
    X_nuevo = np.sign(np.dot(W, X_actual))
    
    # Hopfield convencional define que si el producto es 0, se mantiene el estado previo
    X_nuevo[X_nuevo == 0] = X_actual[X_nuevo == 0]
    
    # Verificar si la red ya se estabilizó
    if np.array_equal(X_nuevo, X_actual):
        print(f"\n¡La red ha convergido en la iteración {i+1}!")
        break
    X_actual = X_nuevo

print("\n--- ESTADO FINAL (IMAGEN RECONOCIDA) ---")
print(X_actual.reshape(4, 4))

# Comprobación de éxito
if np.array_equal(X_actual, X_ideal):
    print("\nResultado: Éxito. Imagen identificada correctamente como el patrón original.")
else:
    print("\nResultado: No convergió al patrón exacto.")