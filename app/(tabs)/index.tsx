import ParallaxScrollView from '@/components/ParallaxScrollView'
import '@tensorflow/tfjs-react-native'
import * as tf from '@tensorflow/tfjs'
import { Platform } from 'react-native'

import { bundleResourceIO } from '@tensorflow/tfjs-react-native'
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native'
import { useEffect, useRef, useState, useCallback } from 'react'
import { ThemedText } from '@/components/ThemedText'
import {
  Camera,
  CameraType,
  CameraView,
  useCameraPermissions,
} from 'expo-camera'
import imageToTensor from '@/constants/imageToSensor'
// import { CameraType } from 'expo-camera/legacy'

// Initialize TensorFlow platform outside component to avoid re-initialization
let tfInitialized = false
let modelCache: tf.LayersModel | null = null

const initializeTensorFlow = async () => {
  if (tfInitialized) return true

  try {
    // Platform-specific initialization
    if (Platform.OS === 'ios') {
      await tf.ready()
    } else {
      // Android optimization
      await tf.ready()
    }

    tfInitialized = true
    console.log('TensorFlow initialized successfully')
    return true
  } catch (error) {
    console.error('TensorFlow initialization failed:', error)
    return false
  }
}

const loadModelAsync = async (onProgress?: (progress: number) => void) => {
  if (modelCache) return modelCache

  try {
    onProgress?.(10) // Starting model load

    const modelJson = require('../../assets/model.json')
    const modelWeight = require('../../assets/weights.bin')
    console.log('Loaded model.json:', modelJson)

    // const modelJson = Asset.fromModule(require('../../assets/model.json')).uri
    // const modelWeights = Asset.fromModule(
    //   require('../../assets/weights.bin')
    // ).uri

    onProgress?.(30) // Assets loaded

    const loadedModel = await tf.loadLayersModel(
      bundleResourceIO(modelJson, modelWeight)
    )

    onProgress?.(80) // Model loaded

    // Warm up the model with a dummy prediction to improve first inference speed
    const dummyInput = tf.zeros([1, 224, 224, 3]) // Adjust dimensions to match your model
    await loadedModel.predict(dummyInput)
    dummyInput.dispose()

    onProgress?.(100) // Model ready

    modelCache = loadedModel
    console.log('Model loaded and warmed up successfully')
    return loadedModel
  } catch (error) {
    console.error('Model loading failed:', error)
    throw error
  }
}

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions()
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [loadingStage, setLoadingStage] = useState('Initializing...')
  const [isReady, setIsReady] = useState(false)
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [facing, setFacing] = useState<CameraType>('back')
  const [analyzing, setAnalyzing] = useState<boolean>(false)
  const [predictionResult, setPredictionResult] = useState<string | null>(null)

  const classNames = ['Positive', 'Negative', 'Invalid']

  const cameraRef = useRef<CameraView>(null)

  const initializeApp = useCallback(async () => {
    try {
      setLoadingStage('Initializing TensorFlow...')
      setLoadingProgress(5)

      const tfReady = await initializeTensorFlow()
      if (!tfReady) {
        throw new Error('Failed to initialize TensorFlow')
      }

      setLoadingProgress(20)
      setLoadingStage('Loading AI Model...')

      const loadedModel = await loadModelAsync((progress) => {
        setLoadingProgress(20 + progress * 0.8) // Scale progress from 20-100
        if (progress === 30) setLoadingStage('Processing model assets...')
        if (progress === 80) setLoadingStage('Warming up model...')
        if (progress === 100) setLoadingStage('Ready!')
      })

      setModel(loadedModel)
      setIsReady(true)
      setLoadingProgress(100)
    } catch (error: any) {
      console.error('Initialization error:', error)
      setError(error.message || 'Failed to initialize app')
    }
  }, [])

  useEffect(() => {
    initializeApp()
  }, [initializeApp])

  // Error state
  if (error) {
    return (
      <View style={styles.centerContainer}>
        <ThemedText style={styles.errorText}>❌ {error}</ThemedText>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => {
            setError(null)
            setLoadingProgress(0)
            setIsReady(false)
            initializeApp()
          }}
        >
          <Text style={styles.buttonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    )
  }

  // Permission loading
  if (!permission) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size='large' color='#4caf50' />
        <ThemedText>Loading camera permissions...</ThemedText>
      </View>
    )
  }

  // Permission denied
  if (!permission.granted) {
    return (
      <View style={styles.centerContainer}>
        <ThemedText style={styles.permissionText}>
          📷 Camera access is required for AI analysis
        </ThemedText>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    )
  }

  // Loading state with progress
  if (!isReady || !model) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size='large' color='#4caf50' />
        <ThemedText style={styles.loadingTitle}>Setting up AI...</ThemedText>
        <ThemedText style={styles.loadingStage}>{loadingStage}</ThemedText>
        <View style={styles.progressContainer}>
          <View
            style={[styles.progressBar, { width: `${loadingProgress}%` }]}
          />
        </View>
        <ThemedText style={styles.progressText}>
          {Math.round(loadingProgress)}%
        </ThemedText>
      </View>
    )
  }

  const handleCapture = async () => {
    if (cameraRef.current && model) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 1,
          base64: true,
        })
        console.log('Photo captured:', photo.uri)
        setAnalyzing(true)

        // Let the UI update first
        setTimeout(async () => {
          try {
            if (!photo.base64) throw new Error('Base64 not available')

            const imageTensor = await imageToTensor(photo.base64)
            const prediction = model.predict(imageTensor) as tf.Tensor
            const result = await prediction.data()
            const predictedClass = result.indexOf(Math.max(...result))

            console.log('Prediction result:', result)

            imageTensor.dispose()
            prediction.dispose()

            setPredictionResult(classNames[predictedClass])
          } catch (error) {
            console.log('Error processing image:', error)
            setPredictionResult('Error: Failed to process image.')
          } finally {
            setAnalyzing(false)
          }
        }, 50) // 50ms delay to allow UI update
      } catch (error) {
        console.log('Error capturing photo:', error)
        setPredictionResult('Error: Failed to capture photo.')
        setAnalyzing(false)
      }
    }
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === 'back' ? 'front' : 'back'))
  }

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#A1CEDC', dark: '#1D3D47' }}
    >
      {/* <View style={{ flex: 1, position: 'relative' }}> */}
      <CameraView style={styles.camera} facing={facing} ref={cameraRef} />
      <TouchableOpacity
        style={styles.captureButton}
        onPress={toggleCameraFacing}
      >
        <ThemedText style={styles.progressText}>📷 Flip Camera</ThemedText>
      </TouchableOpacity>
      <View style={styles.overlay}>
        <ThemedText style={styles.readyText}>🤖 AI Ready</ThemedText>
      </View>
      <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
        <ThemedText style={styles.progressText}>📸 Analyze</ThemedText>
      </TouchableOpacity>

      {(analyzing || predictionResult) && (
        <View style={styles.analyzingModal}>
          {analyzing ? (
            <>
              <ActivityIndicator size='small' color='#4caf50' />
              <ThemedText>Analyzing...</ThemedText>
            </>
          ) : (
            <>
              <ThemedText>🧪 Prediction: {predictionResult}</ThemedText>
              <TouchableOpacity onPress={() => setPredictionResult(null)}>
                <ThemedText>Close</ThemedText>
              </TouchableOpacity>
            </>
          )}
        </View>
      )}
    </ParallaxScrollView>
  )
}

export const styles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: { width: '100%', height: 300, borderWidth: 1, marginTop: 50 },
  overlay: {
    position: 'absolute',
    top: 81,
    left: 34,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    borderRadius: 8,
  },
  readyText: {
    color: '#4caf50',
    fontSize: 16,
    fontWeight: 'bold',
  },
  captureButton: {
    alignSelf: 'center',
    backgroundColor: '#4caf50',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  button: {
    backgroundColor: '#4caf50',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 20,
  },
  retryButton: {
    backgroundColor: '#f44336',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  loadingTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
  },
  loadingStage: {
    fontSize: 14,
    opacity: 0.7,
    marginBottom: 20,
  },
  progressContainer: {
    width: 200,
    height: 6,
    backgroundColor: '#e0e0e0',
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 10,
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#4caf50',
    borderRadius: 3,
  },
  progressText: {
    fontSize: 12,
    opacity: 0.7,
  },
  permissionText: {
    textAlign: 'center',
    fontSize: 16,
    marginBottom: 20,
  },
  errorText: {
    color: '#f44336',
    textAlign: 'center',
    fontSize: 16,
    marginBottom: 20,
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  analyzingModal: {
    position: 'absolute',
    width: '100%',
    height: 80,
    top: '80%',
    left: '10%',
    transform: [{ translateY: '-80%' }],
    backgroundColor: '#e8e8e829',
    borderColor: '#505050',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
})
