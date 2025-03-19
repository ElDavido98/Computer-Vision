import cv2
import sqlite3
from google.colab import drive
import json


drive.mount('/content/drive')


def data_pre_processing():
    for i in range(769, 786):
        if 774 >= i >= 772:
            continue
        time = []
        j = 0
        stop = 0
        last_t = 0

        # Create file path
        dbfile = f"drive/MyDrive/Computer_Vision/intersection_dbfiles/intsc_data_{i}.db"
        # Create a SQL connection to our SQLite database
        con = sqlite3.connect(dbfile)
        # Create cursor
        cur = con.cursor()

        elements = cur.execute("SELECT * FROM 'TRAFFIC_LIGHTS'").fetchall()

        # Open video once
        vidcap = cv2.VideoCapture(f"drive/MyDrive/Computer_Vision/annotated_videos/{i}.avi")
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        with open(f"file_{i}.json", "w") as f:
            f.write("[")  # Start JSON array
            first_entry = True

            while not stop:
                seconds = round(0.4004 * j, 4)
                t_msec = 1000 * seconds
                vidcap.set(cv2.CAP_PROP_POS_MSEC, t_msec)
                ret, frame = vidcap.read()

                # Check if ret is False, indicating the end of the video
                if not ret:
                    stop = 1
                else:
                    # 'frame' is a numpy.ndarray
                    frame = cv2.resize(frame, (160, 90))  # Smaller size to save memory
                    scene_data = frame.tolist()  # Optional: Compress further using an image library

                    traffic_data = None
                    for t in range(last_t, len(elements)):
                        if seconds >= elements[t][-1]:
                            traffic_data = list(elements[t][1:-1])
                            last_t = t
                            break

                    # Take all the informations of time "seconds"
                    query = f"SELECT TRACK_ID, X, Y, SPEED, TAN_ACC, LAT_ACC, ANGLE, TIME FROM 'TRAJECTORIES_0{i}' WHERE TIME = {seconds}"
                    seq = cur.execute(query).fetchall()

                    entry = {"video_id": i, "time": seconds, "scene_data": scene_data, "trajectories": seq,
                             "traffic_data": traffic_data}
                    if not first_entry:
                        f.write(",")  # Add comma between JSON objects
                    json.dump(entry, f)
                    first_entry = False

                j += 1

            f.write("]")  # End JSON array

        vidcap.release()
        con.close()
        print(f"Video {i} processing completed.")
