from Implementations import implementations


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    teller = 0
    for minibatch_y, minibatch_tx in implementations.batch_iter(y, tx, batch_size, max_iters):
        gradient = implementations.compute_gradient(minibatch_y, minibatch_tx, w)
        loss = implementations.compute_loss(minibatch_y, minibatch_tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=teller, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        teller += 1
    return losses, ws


