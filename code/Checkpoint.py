import tensorflow as tf
import os.path

def restore_from_checkpoint(sess, saver, ckpt_dir, step=None):
    #print("Trying to restore from checkpoint in dir", ckpt_dir, "at step", step)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (step == None):
        if ckpt and ckpt.model_checkpoint_path:
            print("Checkpoint file is ", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            #print("Restored from checkpoint %s" % global_step)
            return global_step
        else:
            print("No checkpoint file found")
            assert False
    else:
        #print("ckpt: ", ckpt)

        path = ckpt.model_checkpoint_path[0: ckpt.model_checkpoint_path.index('-') + 1] + str(step)

        #print("path: ",path)
        saver.restore(sess, path)
        #print("Restored from checkpoint %d" % step)
        return step

def optionally_restore_from_checkpoint(sess, saver, train_dir):
    while True:
        response = input("Restore from checkpoint [y/n]? ").lower()
        if response == 'y':
            return restore_from_checkpoint(sess, saver, train_dir)
        if response == 'n':
            return 0

