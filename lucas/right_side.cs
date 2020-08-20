        RightArm = humanoid.transform.Find("Hips/Spine/Spine1/Spine2/RightClav/RightArm");
        Transform RightForeArm = RightArm.Find("RightForeArm");
        Transform RightHand = RightForeArm.Find("RightHand");
        Transform RightHandWrist2 = RightHand.Find("RightHandWrist2");
        // Index
        Transform RightHandIndex1 = RightHandWrist2.Find("RightHandIndex1");
        Transform RightHandIndex2 = RightHandIndex1.Find("RightHandIndex2");
        Transform RightHandIndex3 = RightHandIndex2.Find("RightHandIndex3");

        // Middle
        Transform RightHandMiddle1 = RightHandWrist2.Find("RightHandMiddle1");
        Transform RightHandMiddle2 = RightHandMiddle1.Find("RightHandMiddle2");
        Transform RightHandMiddle3 = RightHandMiddle2.Find("RightHandMiddle3");

        // Ring
        Transform RightHandRing1 = RightHandWrist2.Find("RightHandRing1");
        Transform RightHandRing2 = RightHandRing1.Find("RightHandRing2");
        Transform RightHandRing3 = RightHandRing2.Find("RightHandRing3");

        // Pinky
        Transform RightHandMetacarpal = RightHandWrist2.Find("RightHandMetacarpal");
        Transform RightHandPinky1 = RightHandMetacarpal.Find("RightHandPinky1");
        Transform RightHandPinky2 = RightHandPinky1.Find("RightHandPinky2");
        Transform RightHandPinky3 = RightHandPinky2.Find("RightHandPinky3");

        // Thumb
        Transform RightHandThumb1 = RightHandWrist2.Find("RightHandThumb1");
        Transform RightHandThumb2 = RightHandThumb1.Find("RightHandThumb2");
        Transform RightHandThumb3 = RightHandThumb2.Find("RightHandThumb3");

        jointToTransform.Add("RightArm", RightArm);
        jointToTransform.Add("RightForeArm", RightForeArm);
        jointToTransform.Add("RightHand", RightHand);
        jointToTransform.Add("RightHandWrist2", RightHandWrist2);

        //Index
        jointToTransform.Add("RightHandIndex1", RightHandIndex1);
        jointToTransform.Add("RightHandIndex2", RightHandIndex2);
        jointToTransform.Add("RightHandIndex3", RightHandIndex3);

        // Middle
        jointToTransform.Add("RightHandMiddle1", RightHandMiddle1);
        jointToTransform.Add("RightHandMiddle2", RightHandMiddle2);
        jointToTransform.Add("RightHandMiddle3", RightHandMiddle3);

        // Ring
        jointToTransform.Add("RightHandRing1", RightHandRing1);
        jointToTransform.Add("RightHandRing2", RightHandRing2);
        jointToTransform.Add("RightHandRing3", RightHandRing3);

        // Pinky
        jointToTransform.Add("RightHandMetacarpal", RightHandMetacarpal);
        jointToTransform.Add("RightHandPinky1", RightHandPinky1);
        jointToTransform.Add("RightHandPinky2", RightHandPinky2);
        jointToTransform.Add("RightHandPinky3", RightHandPinky3);

        // Thumb
        jointToTransform.Add("RightHandThumb1", RightHandThumb1);
        jointToTransform.Add("RightHandThumb2", RightHandThumb2);
        jointToTransform.Add("RightHandThumb3", RightHandThumb3);