import dlib
import cv2
import numpy as np

from pedestrian.tracking.DistanceConnector import DistanceConnector
from pedestrian.tracking.Track import Track
from pedestrian.tracking.Tracker import Tracker


class Centroid(Tracker):

    __slots__ = ["frame_count", "det_period", "trackers", "min_score", "next_idx"]

    def __init__(self, det_period: int):
        self.frame_count = 0
        self.det_period = det_period
        self.trackers = []
        self.tracks = dict()
        self.next_idx = 1

    def track(self, frame, dets):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if self.frame_count % self.det_period == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trks = np.zeros((len(self.tracks), 5)) # [x1, y1, x2, y2, idx]

            for i, (idx, t) in self.tracks.items():
                (x1, y1, x2, y2) = t[-1, :]
                trks[i, :-1] = [x1, y1, x2, y2, idx]

            matches = DistanceConnector().connect(dets, trks)

            # loop over the detections
            for i in np.arange(0, dets.shape[0]):
                if i not in matches[:, 0]:
                    box = dets[i, :]
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    self.trackers.append(tracker)
                    det =
                else:
                    (det, idx, _) = matches[matches[:, 0] == i, :]
                    track = self.tracks.get(idx, Track(idx, np.empty((0, 4))))
                    track.add(np.array([x1, y1, x2, y2]))
                    self.tracks[idx] = track
                    self.tracks[trk].add(dets[det, :])


        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in self.trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()
