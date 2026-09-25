import React, { useState, useRef } from "react";
import {
  StyleSheet,
  View,
  ActivityIndicator,
  Text,
  StatusBar,
  Platform,
  BackHandler,
  TouchableOpacity,
  Linking,
  LogBox,
  Alert,
} from "react-native";
import {
  SafeAreaView,
  SafeAreaProvider,
  useSafeAreaInsets,
} from "react-native-safe-area-context";
import { WebView } from "react-native-webview";
import * as ImagePicker from "expo-image-picker";
import Constants from "expo-constants";
import {
  useAudioRecorder,
  useAudioRecorderState,
  AudioModule,
  RecordingPresets,
  setAudioModeAsync,
} from "expo-audio";
import { File } from "expo-file-system";

// Ignore transient dev-server reconnection and URL scheme warnings
LogBox.ignoreLogs([
  "Cannot connect to Expo CLI",
  "net::ERR_UNKNOWN_URL_SCHEME",
  "WebView error: net::ERR_UNKNOWN_URL_SCHEME",
]);

/**
 * Resolve the dev-machine's web app URL.
 *
 * In normal (LAN) mode Expo's debugger host is the machine's LAN IP, and the
 * WebView can load the web app from http://<LAN-IP>:5173.
 *
 * In tunnel mode (npx expo start --tunnel) the Expo host becomes an
 * "exp.direct" domain that only forwards Metro on :8081 — NOT the web app on
 * :5173. In that case we fall back to the LAN IP, which means the phone must
 * be on the same Wi-Fi as this machine for the WebView to load the web UI.
 */
function getWebAppUrl(): string {
  // Explicit override: when the web app is served through a public tunnel
  // (Expo's --tunnel only forwards Metro on :8081), point the WebView at the
  // public URL of the Vite app instead of the LAN IP.
  const webTunnelUrl = process.env.EXPO_PUBLIC_WEB_URL;
  if (webTunnelUrl) {
    return webTunnelUrl;
  }
  const debuggerHost =
    Constants.expoConfig?.hostUri ?? Constants.manifest2?.extra?.expoGo?.debuggerHost;
  if (debuggerHost && !debuggerHost.includes("exp.direct")) {
    const ip = debuggerHost.split(":")[0];
    if (ip && ip !== "localhost" && ip !== "127.0.0.1") {
      return `http://${ip}:5173`;
    }
  }
  // Fallback — current LAN IP (also used when Expo runs in tunnel mode)
  return "http://10.215.181.180:5173";
}

const WEB_APP_URL = getWebAppUrl();

function formatVoiceTime(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppInner />
    </SafeAreaProvider>
  );
}

function AppInner() {
  const webViewRef = useRef<WebView>(null);
  const insets = useSafeAreaInsets();
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [canGoBack, setCanGoBack] = useState(false);
  const [captureBusy, setCaptureBusy] = useState(false);

  // ─── Native voice recording (expo-audio, included in Expo Go) ─────────────
  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(audioRecorder);
  const [voiceRecording, setVoiceRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);

  /** Deliver a recorded voice note (or a cancel) back into the web app. */
  const injectNativeVoice = (
    payload:
      | { dataUrl: string; mimeType: string; durationMs: number }
      | null,
  ) => {
    const js = payload
      ? `(function(){
          try {
            if (window.__simplificantNativeBridge && window.__simplificantNativeBridge.onNativeVoice) {
              window.__simplificantNativeBridge.onNativeVoice(${JSON.stringify(payload)});
            }
          } catch (e) { console.error('nativeVoiceBridge', e); }
          true;
        })();`
      : `(function(){
          try {
            if (window.__simplificantNativeBridge && window.__simplificantNativeBridge.onNativeVoiceCancelled) {
              window.__simplificantNativeBridge.onNativeVoiceCancelled();
            }
          } catch (e) { console.error('nativeVoiceBridge', e); }
          true;
        })();`;
    webViewRef.current?.injectJavaScript(js);
  };

  const startNativeVoiceRecording = async () => {
    console.log("NATIVE: voice recording requested");
    if (voiceBusy || voiceRecording) return;
    setVoiceBusy(true);
    try {
      const perm = await AudioModule.requestRecordingPermissionsAsync();
      if (!perm.granted) {
        console.log("NATIVE: microphone permission denied");
        Alert.alert(
          "Microphone permission needed",
          "Allow microphone access so you can record voice notes.",
        );
        injectNativeVoice(null);
        return;
      }
      console.log("NATIVE: microphone permission granted");
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await audioRecorder.prepareToRecordAsync();
      audioRecorder.record();
      console.log("NATIVE: recording started");
      setVoiceRecording(true);
    } catch (err) {
      console.warn("startNativeVoiceRecording failed", err);
      injectNativeVoice(null);
    } finally {
      setVoiceBusy(false);
    }
  };

  const finishNativeVoiceRecording = async (sendToWeb: boolean) => {
    console.log("NATIVE: recording stopped");
    const wasRecording = voiceRecording;
    setVoiceRecording(false);
    try {
      if (wasRecording || recorderState.isRecording) {
        await audioRecorder.stop();
      }
    } catch (err) {
      console.warn("stop native voice recording failed", err);
    }
    if (!sendToWeb) {
      injectNativeVoice(null);
      return;
    }
    const uri = audioRecorder.uri;
    if (!uri) {
      injectNativeVoice(null);
      return;
    }
    try {
      const file = new File(uri);
      const base64 = await file.base64();
      injectNativeVoice({
        dataUrl: `data:audio/mp4;base64,${base64}`,
        mimeType: "audio/mp4",
        durationMs: Math.round(recorderState.durationMillis ?? 0),
      });
      console.log("NATIVE: audio returned to web");
    } catch (err) {
      console.warn("read native voice recording failed", err);
      injectNativeVoice(null);
    }
  };

  // Auto-stop after 60 s so an unattended recording can't run away.
  React.useEffect(() => {
    if (voiceRecording && (recorderState.durationMillis ?? 0) >= 60000) {
      finishNativeVoiceRecording(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voiceRecording, recorderState.durationMillis]);

  // Stop any in-flight recording if the app shell unmounts.
  React.useEffect(() => {
    return () => {
      if (audioRecorder.isRecording) {
        audioRecorder.stop().catch(() => {});
      }
    };
  }, [audioRecorder]);

  /**
   * Deliver an image captured natively (camera or gallery) into the web app.
   * The web app registers `window.__simplificantNativeBridge` once it boots;
   * if it isn't ready yet we stash the image on `__simplificantPendingImage`
   * and the web app drains it when it mounts.
   *
   * Android WebView's evaluateJavascript has a practical size limit — a single
   * injectJavaScript call carrying a full-res base64 photo can crash the
   * WebView renderer, which makes the page reload ("app reloads after OK").
   * So the payload is pushed in small chunks and reassembled on the page.
   */
  const CHUNK_SIZE = 128_000; // chars of base64 per injectJavaScript call

  const injectNativeImage = (asset: ImagePicker.ImagePickerAsset) => {
    if (!asset.base64) {
      Alert.alert(
        "Photo unavailable",
        "Could not read the image. Please try again.",
      );
      return;
    }
    const webview = webViewRef.current;
    if (!webview) return;
    const dataUrl = `data:image/jpeg;base64,${asset.base64}`;

    const chunks: string[] = [];
    for (let i = 0; i < dataUrl.length; i += CHUNK_SIZE) {
      chunks.push(dataUrl.slice(i, i + CHUNK_SIZE));
    }

    webview.injectJavaScript(
      "(function(){ window.__simplificantPhotoChunks = []; true; })();",
    );
    for (const chunk of chunks) {
      webview.injectJavaScript(
        `(function(){ window.__simplificantPhotoChunks.push(${JSON.stringify(chunk)}); true; })();`,
      );
    }
    webview.injectJavaScript(
      `(function(){
        try {
          var parts = window.__simplificantPhotoChunks || [];
          window.__simplificantPhotoChunks = [];
          var url = parts.join('');
          if (window.__simplificantNativeBridge && window.__simplificantNativeBridge.onNativeImage) {
            window.__simplificantNativeBridge.onNativeImage(url);
          } else {
            window.__simplificantPendingImage = url;
          }
        } catch (e) { console.error('nativeImageBridge', e); }
        true;
      })();`,
    );
  };

  const handleTakePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert(
        "Camera permission needed",
        "Allow camera access so you can photograph your crafts.",
      );
      return;
    }
    setCaptureBusy(true);
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ["images"],
        quality: 0.5,
        exif: false,
        base64: true,
        allowsEditing: false,
      });
      if (!result.canceled && result.assets?.[0]) {
        injectNativeImage(result.assets[0]);
      }
    } catch (err) {
      console.warn("launchCameraAsync failed", err);
    } finally {
      setCaptureBusy(false);
    }
  };

  const handleChooseFromGallery = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert(
        "Photos permission needed",
        "Allow photo access so you can pick craft images from your gallery.",
      );
      return;
    }
    setCaptureBusy(true);
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["images"],
        quality: 0.5,
        exif: false,
        base64: true,
        allowsEditing: false,
      });
      if (!result.canceled && result.assets?.[0]) {
        injectNativeImage(result.assets[0]);
      }
    } catch (err) {
      console.warn("launchImageLibraryAsync failed", err);
    } finally {
      setCaptureBusy(false);
    }
  };

  const openNativeCaptureMenu = () => {
    Alert.alert(
      "Add Craft Photo",
      "Capture your craft with the camera or pick one from your gallery.",
      [
        { text: "📷 Take Photo", onPress: handleTakePhoto },
        { text: "🖼 Choose from Gallery", onPress: handleChooseFromGallery },
        { text: "Cancel", style: "cancel" },
      ],
    );
  };

  // Android hardware back button → navigate back in WebView
  React.useEffect(() => {
    if (Platform.OS !== "android") return;

    const onBackPress = () => {
      // While recording, back should not dismiss the recording overlay.
      if (voiceRecording) return true;
      if (canGoBack && webViewRef.current) {
        webViewRef.current.goBack();
        return true;
      }
      return false;
    };

    const subscription = BackHandler.addEventListener("hardwareBackPress", onBackPress);
    return () => subscription.remove();
  }, [canGoBack, voiceRecording]);

  // JavaScript injected into the WebView to make the web app feel native
  const INJECTED_JS = `
    (function() {
      document.addEventListener('contextmenu', function(e) { e.preventDefault(); });
      var meta = document.querySelector('meta[name="viewport"]');
      if (!meta) {
        meta = document.createElement('meta');
        meta.name = 'viewport';
        document.head.appendChild(meta);
      }
      meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
      true;
    })();
  `;

  if (hasError) {
    return (
      <SafeAreaProvider>
        <SafeAreaView style={styles.errorContainer}>
          <StatusBar barStyle="light-content" backgroundColor="#241C15" />
          <Text style={styles.errorEmoji}>🔌</Text>
          <Text style={styles.errorTitle}>Cannot Connect</Text>
          <Text style={styles.errorMessage}>
            Make sure the web app dev server is running{"\n"}
            <Text style={styles.errorUrl}>{WEB_APP_URL}</Text>
          </Text>
          <Text style={styles.errorHint}>
            Run <Text style={styles.errorCode}>npm run web:dev</Text> from the project root
          </Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={() => {
              setHasError(false);
              setIsLoading(true);
            }}
          >
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FDFBF7" />

        {isLoading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#C9922E" />
            <Text style={styles.loadingText}>Loading Simplificant...</Text>
            <Text style={styles.loadingUrl}>{WEB_APP_URL}</Text>
          </View>
        )}

        <WebView
          ref={webViewRef}
          source={{ uri: WEB_APP_URL }}
          style={styles.webview}
          javaScriptEnabled={true}
          domStorageEnabled={true}
          startInLoadingState={false}
          scalesPageToFit={true}
          allowsFullscreenVideo={true}
          mediaPlaybackRequiresUserAction={false}
          allowsInlineMediaPlayback={true}
          mediaCapturePermissionGrantType="grantIfSameHostElsePrompt"
          allowFileAccess={true}
          onNavigationStateChange={(navState) => {
            setCanGoBack(navState.canGoBack);
          }}
          onLoadStart={() => setIsLoading(true)}
          onLoadEnd={() => setIsLoading(false)}
          onError={(syntheticEvent) => {
            const { nativeEvent } = syntheticEvent;
            // Ignore custom protocol schemes (tel:, mailto:, sms:, whatsapp:, etc.) - do not show fatal error screen
            if (
              nativeEvent.description?.includes("ERR_UNKNOWN_URL_SCHEME") ||
              nativeEvent.description?.includes("net::ERR_UNKNOWN_URL_SCHEME")
            ) {
              setIsLoading(false);
              return;
            }
            console.warn("WebView error:", nativeEvent.description);
            setIsLoading(false);
            setHasError(true);
          }}
          onHttpError={(syntheticEvent) => {
            const { statusCode } = syntheticEvent.nativeEvent;
            if (statusCode >= 500) {
              setIsLoading(false);
              setHasError(true);
            }
          }}
          onShouldStartLoadWithRequest={(request) => {
            const url = request.url;
            // Allow normal web navigation inside WebView
            if (
              url.startsWith("http://") ||
              url.startsWith("https://") ||
              url.startsWith("about:blank") ||
              url.startsWith("data:") ||
              url.startsWith("blob:")
            ) {
              return true;
            }
            // Delegate external schemes (tel:, mailto:, sms:, whatsapp:, etc.) to device system apps
            Linking.canOpenURL(url)
              .then((supported) => {
                if (supported) {
                  Linking.openURL(url).catch(() => {});
                }
              })
              .catch(() => {});
            return false;
          }}
          injectedJavaScript={INJECTED_JS}
          onMessage={(event) => {
            // Commands from the web app's speechService (Web Speech API is
            // missing in Android WebView, so it delegates to this recorder)
            // and from the web UI's "Open Live Camera" / gallery buttons.
            try {
              const data = JSON.parse(event.nativeEvent.data);
              if (data?.type === "voice:start") {
                startNativeVoiceRecording();
              } else if (data?.type === "voice:stop") {
                finishNativeVoiceRecording(true);
              } else if (data?.type === "camera:take") {
                handleTakePhoto();
              } else if (data?.type === "camera:gallery") {
                handleChooseFromGallery();
              }
            } catch (err) {
              console.warn("voice message parse failed", err);
            }
          }}
          pullToRefreshEnabled={true}
          bounces={true}
          mixedContentMode="always"
          applicationNameForUserAgent="SimplificantMobile/1.0"
          originWhitelist={["*"]}
          setSupportMultipleWindows={false}
          cacheEnabled={true}
        />

        {/* Native voice recording overlay. Android WebView has no Web Speech
            API, so the shell records real audio here (expo-audio) and injects
            it into the web app as a voice note via the bridge. */}
        {voiceRecording && (
          <View style={styles.voiceOverlay}>
            <View style={styles.voiceCard}>
              <Text style={styles.voiceMic}>🎙️</Text>
              <Text style={styles.voiceTitle}>Recording voice note…</Text>
              <Text style={styles.voiceTimer}>
                {formatVoiceTime(recorderState.durationMillis ?? 0)}
              </Text>
              <TouchableOpacity
                style={styles.voiceStopButton}
                activeOpacity={0.85}
                onPress={() => finishNativeVoiceRecording(true)}
              >
                <Text style={styles.voiceStopText}>⏹ Stop &amp; Send</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.voiceCancelButton}
                activeOpacity={0.85}
                onPress={() => finishNativeVoiceRecording(false)}
              >
                <Text style={styles.voiceCancelText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Floating native capture button. The web app's live viewfinder is
            unavailable in a WebView over HTTP (getUserMedia), so photos are
            taken with the device camera / picked from the gallery natively
            and injected into the web app via __simplificantNativeBridge. */}
        <View
          style={[styles.captureFabWrap, { bottom: insets.bottom + 100 }]}
          pointerEvents="box-none"
        >
          <TouchableOpacity
            style={styles.captureFab}
            activeOpacity={0.85}
            onPress={openNativeCaptureMenu}
            disabled={captureBusy}
          >
            <Text style={styles.captureFabText}>
              {captureBusy ? "⏳ Adding…" : "📷 Add Photo"}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FDFBF7",
  },
  webview: {
    flex: 1,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
    backgroundColor: "#FDFBF7",
    alignItems: "center",
    justifyContent: "center",
  },
  loadingText: {
    marginTop: 16,
    fontSize: 17,
    fontWeight: "600",
    color: "#241C15",
  },
  loadingUrl: {
    marginTop: 6,
    fontSize: 12,
    color: "#9C9182",
  },
  errorContainer: {
    flex: 1,
    backgroundColor: "#241C15",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 32,
  },
  errorEmoji: {
    fontSize: 64,
    marginBottom: 16,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#FDFBF7",
    marginBottom: 12,
  },
  errorMessage: {
    fontSize: 15,
    color: "#B0A489",
    textAlign: "center",
    lineHeight: 22,
  },
  errorUrl: {
    color: "#C9922E",
    fontWeight: "600",
  },
  errorHint: {
    marginTop: 20,
    fontSize: 13,
    color: "#6B6255",
    textAlign: "center",
  },
  errorCode: {
    color: "#DCA33C",
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
  },
  retryButton: {
    marginTop: 28,
    backgroundColor: "#C9922E",
    paddingHorizontal: 36,
    paddingVertical: 14,
    borderRadius: 12,
  },
  retryText: {
    color: "#FDFBF7",
    fontSize: 16,
    fontWeight: "700",
  },
  captureFabWrap: {
    position: "absolute",
    right: 14,
    zIndex: 20,
    elevation: 8,
  },
  captureFab: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#241C15",
    borderColor: "#C9922E",
    borderWidth: 1.5,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 999,
    shadowColor: "#000",
    shadowOpacity: 0.28,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
    elevation: 8,
  },
  captureFabText: {
    color: "#F7F2E9",
    fontSize: 14,
    fontWeight: "700",
  },
  voiceOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 30,
    elevation: 10,
    backgroundColor: "rgba(24, 18, 12, 0.72)",
    alignItems: "center",
    justifyContent: "center",
  },
  voiceCard: {
    width: "78%",
    maxWidth: 320,
    backgroundColor: "#241C15",
    borderColor: "#C9922E",
    borderWidth: 1.5,
    borderRadius: 20,
    paddingVertical: 28,
    paddingHorizontal: 24,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.4,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 6 },
    elevation: 10,
  },
  voiceMic: {
    fontSize: 52,
  },
  voiceTitle: {
    marginTop: 10,
    fontSize: 17,
    fontWeight: "700",
    color: "#F7F2E9",
  },
  voiceTimer: {
    marginTop: 6,
    fontSize: 28,
    fontVariant: ["tabular-nums"],
    color: "#C9922E",
    fontWeight: "600",
  },
  voiceStopButton: {
    marginTop: 22,
    backgroundColor: "#C0392B",
    paddingHorizontal: 34,
    paddingVertical: 12,
    borderRadius: 999,
  },
  voiceStopText: {
    color: "#FDFBF7",
    fontSize: 15,
    fontWeight: "700",
  },
  voiceCancelButton: {
    marginTop: 12,
    paddingHorizontal: 24,
    paddingVertical: 8,
  },
  voiceCancelText: {
    color: "#B0A489",
    fontSize: 14,
    fontWeight: "600",
  },
});
