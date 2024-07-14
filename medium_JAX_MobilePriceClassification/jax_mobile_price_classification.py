import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#%% import data and prepare a subset

# train.csv is split into training and test datasets (test.csv is not used for now)
df = pd.read_csv('data/train.csv') # 2000 rows x 21 columns
df = df.iloc[:, 10:] # select last 11 columns (10 features + 1 target class)
df = df.loc[df['price_range'] <= 1] # select rows where 'price_range' is 0 or 1
print(df.head())

#%% prepare training & test datasets as JAX arrays

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.20,
                                                    stratify= y)

X_train, X_test, y_train, y_test = jnp.array(X_train), jnp.array(X_test), \
                                    jnp.array(y_train), jnp.array(y_test)
                                    
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%% build a Logistic Regression model

# sigmoid activation function
def activation(r):
    return 1 / (1 + jnp.exp(-r))

# binary loss function (since we have two classes)
def loss(bias, weight, X, y, lmbd=0.1):
    # lmbd: L2-regularization constant
    a = activation(jnp.dot(X, weight) + bias)
    loss = jnp.sum(y * jnp.log(a) + (1-y) * jnp.log(1-a)) / y.size
    regularization = 0.5 * lmbd * (jnp.dot(weight, weight) + bias * bias)
    return -loss + regularization

#%% training loop

num_epochs, lr = 1000, 1e-2 
w = 1e-5 * jnp.ones(X.shape[1]) # 1 x 10-array
b = 1.0
history = [float(loss(b, w, X_train, y_train))]
for e in range(num_epochs):
    b_old = b
    b -= lr * jax.grad(loss, argnums=0)(b_old, w, X_train, y_train)
    w -= lr * jax.grad(loss, argnums=1)(b_old, w, X_train, y_train)
    history.append(float(loss(b, w, X_train, y_train)))
print(history)
    
# plot the Loss-history
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, num_epochs+1), history, 'b-')

# Set the title and labels
plt.title('Training History', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Customize the tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

#%% test the model using trained model parameters (w, b)

y_pred = jnp.array( activation( jnp.dot(X_test, w) + b) )
y_pred = jnp.where(y_pred > 0.5, 1, 0) 
print(classification_report(y_test, y_pred))
    
    
    
