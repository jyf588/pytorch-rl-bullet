        LeftArm = humanoid.transform.Find("Hips/Spine/Spine1/Spine2/LeftClav/LeftArm");
        Transform LeftForeArm = LeftArm.Find("LeftForeArm");
        Transform LeftHand = LeftForeArm.Find("LeftHand");
        Transform LeftHandWrist2 = LeftHand.Find("LeftHandWrist2");
        // Index
        Transform LeftHandIndex1 = LeftHandWrist2.Find("LeftHandIndex1");
        Transform LeftHandIndex2 = LeftHandIndex1.Find("LeftHandIndex2");
        Transform LeftHandIndex3 = LeftHandIndex2.Find("LeftHandIndex3");

        // Middle
        Transform LeftHandMiddle1 = LeftHandWrist2.Find("LeftHandMiddle1");
        Transform LeftHandMiddle2 = LeftHandMiddle1.Find("LeftHandMiddle2");
        Transform LeftHandMiddle3 = LeftHandMiddle2.Find("LeftHandMiddle3");

        // Ring
        Transform LeftHandRing1 = LeftHandWrist2.Find("LeftHandRing1");
        Transform LeftHandRing2 = LeftHandRing1.Find("LeftHandRing2");
        Transform LeftHandRing3 = LeftHandRing2.Find("LeftHandRing3");

        // Pinky
        Transform LeftHandMetacarpal = LeftHandWrist2.Find("LeftHandMetacarpal");
        Transform LeftHandPinky1 = LeftHandMetacarpal.Find("LeftHandPinky1");
        Transform LeftHandPinky2 = LeftHandPinky1.Find("LeftHandPinky2");
        Transform LeftHandPinky3 = LeftHandPinky2.Find("LeftHandPinky3");

        // Thumb
        Transform LeftHandThumb1 = LeftHandWrist2.Find("LeftHandThumb1");
        Transform LeftHandThumb2 = LeftHandThumb1.Find("LeftHandThumb2");
        Transform LeftHandThumb3 = LeftHandThumb2.Find("LeftHandThumb3");

        jointToTransform.Add("LeftArm", LeftArm);
        jointToTransform.Add("LeftForeArm", LeftForeArm);
        jointToTransform.Add("LeftHand", LeftHand);
        jointToTransform.Add("LeftHandWrist2", LeftHandWrist2);

        //Index
        jointToTransform.Add("LeftHandIndex1", LeftHandIndex1);
        jointToTransform.Add("LeftHandIndex2", LeftHandIndex2);
        jointToTransform.Add("LeftHandIndex3", LeftHandIndex3);

        // Middle
        jointToTransform.Add("LeftHandMiddle1", LeftHandMiddle1);
        jointToTransform.Add("LeftHandMiddle2", LeftHandMiddle2);
        jointToTransform.Add("LeftHandMiddle3", LeftHandMiddle3);

        // Ring
        jointToTransform.Add("LeftHandRing1", LeftHandRing1);
        jointToTransform.Add("LeftHandRing2", LeftHandRing2);
        jointToTransform.Add("LeftHandRing3", LeftHandRing3);

        // Pinky
        jointToTransform.Add("LeftHandMetacarpal", LeftHandMetacarpal);
        jointToTransform.Add("LeftHandPinky1", LeftHandPinky1);
        jointToTransform.Add("LeftHandPinky2", LeftHandPinky2);
        jointToTransform.Add("LeftHandPinky3", LeftHandPinky3);

        // Thumb
        jointToTransform.Add("LeftHandThumb1", LeftHandThumb1);
        jointToTransform.Add("LeftHandThumb2", LeftHandThumb2);
        jointToTransform.Add("LeftHandThumb3", LeftHandThumb3);