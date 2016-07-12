import tensorflow as tf;
def lambda_handler(event, context):
    a = tf.constant(event['a'])
    b = tf.constant(event['b'])

    with tf.Session() as sess:
        return str(sess.run(tf.add(a , b)))

if __name__ == '__main__':
    print(lambda_handler({'a' : 10, 'b' : 20}, {}));