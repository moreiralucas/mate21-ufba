import numpy as np

# Ativação do sigmoide
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Derivada da função sigmoide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} finalizada".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Atualiza os pesos e bias da rede aplicando a descida
        de gradiente usando backpropagation para um único mini lote.
        O `mini_batch` é uma lista de tuplas `(x,y)`, e `eta` é a taxa de aprendizado.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
        
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights,nabla_w)]
        self.biases = [b-eta/len(mini_batch)*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Retorna uma tupla `(nabla_b, nabla_w)` representando o
        gradiente para a função de custo C_x. `nabla_b` e
        `nabla_w` são listas de camadas de matrizes numpy, semelhantes
        a `self.biases` e `self.wights`.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x

        # Lista para armazenar todas as ativações, camada por camada
        activations = [x]

        # Lista para armazenar todos os vetores z, camda por camada
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Aqui, l = 1 significa a última camada de neurônios,
        # l = 2 é a segunda e assim por diante
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

if __name__ == '__main__':
    rede1 = Network([2, 3, 1])