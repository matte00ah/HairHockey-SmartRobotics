import rospy
import tf

listener = tf.TransformListener()

target_frame = "world"
source_frame = "tmp0"

try:
    # Aspetta che la trasformazione sia disponibile
    listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))

    # Ottieni la trasformazione (posizione e orientazione)
    # La tupla restituita è (traslazione, rotazione_quaternione)
    (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

    # rot è un quaternione (x, y, z, w)
    # Converti il quaternione in angoli di Eulero (Roll, Pitch, Yaw)
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(rot)

    rospy.loginfo("Orientazione (RPY) di %s rispetto a %s: %.2f, %.2f, %.2f" %
                  (source_frame, target_frame, roll, pitch, yaw))

except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf.TimeoutException) as ex:
    rospy.logerr(ex)