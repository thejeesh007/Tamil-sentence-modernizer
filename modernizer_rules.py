from inltk.inltk import setup, get_similar_sentences, tokenize
from deep_translator import GoogleTranslator
import re
from difflib import SequenceMatcher
import numpy as np

# Setup Tamil language for inltk
try:
    setup('ta')
except:
    pass

# --- 1. Normalize function to match sentences robustly ---
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip())

# --- 2. Word-level classical to modern mappings ---
WORD_MAPPINGS = {
    # Classical words -> Modern equivalents
    "செல்கிறான்": "போறான்",
    "மேற்கொண்ட": "செய்த",
    "சிந்திக்கிறேன்": "யோசிக்கிறேன்",
    "மிகவும்": "ரொம்ப",
    "விழாவிற்கு": "விழாவுக்கு",
    "வந்தாள்": "வந்தா",
    "உரையாற்றுகின்றார்": "பேசுறார்",
    "யாதர்த்தம்": "உண்மை",
    "பேசுகின்றார்": "சொல்றார்",
    "வாழ்நாளில்": "வாழ்க்கையில்",
    "செயல்களைப்பற்றி": "செயல்களைப் பற்றி",
    "கொண்டவன்": "பண்ணுவான்",
    "படிக்கிறான்": "படிக்கிறான்",
    "வாடையும்": "வாடைக் காற்றும்",
    "பிரிந்தினோர்க்கு": "பிரிந்தவர்களுக்கு",
    "அழலே": "நெருப்புபோல இருக்கு",
    "மென்மையாய்": "மென்மையா",
    "இருந்தது": "இருந்தது",
    "பெய்யும்": "வரும்",
    "சுழற்காற்றின்": "காற்று",
    "இசை": "சத்தம்",
    "இனிமையானது": "இனிமையா இருக்கு",
    "விரித்து": "போட்டு",
    "பறந்தான்": "பறந்தது",
    "ஆனந்தம்": "சந்தோஷம்",
    "தரும்": "தருது",
    "இளமை": "யுவதி",
    "பெண்ணாக": "மாதிரி",
    "நடக்கிறாள்": "நடக்கிறா",
    "நிறைந்தது": "நிறைய",
    "அமைதியாக": "அமைதியா",
    "இருந்தது": "இருந்தது",
    "பசிக்கொண்டு": "பசிக்குமே",
    "சாப்பிடவில்லை": "சாப்பிடல",
    "ஓசை": "சத்தம்",
    "இசைவதாக": "சேர்ந்து இனிமையா",
    "மகிழ்ச்சியுடன்": "சந்தோஷமா",
    "சேகரித்தாள்": "எடுத்தா",
    "அறிவுடன்": "புத்திசாலியா",
    "பிரமிக்கச்": "ஆச்சரியப்படுத்த",
    "செய்தான்": "செய்தான்",
    "பக்தி": "மதிப்பு",
    "கொண்டவள்": "கொண்டவள்",
    "ஸ்பரிசம்": "தொடுதல்",
    "தென்றலாக்கியது": "நிம்மதியா இருந்தது",
    "கனவு": "கனவுல",
    "காணும்": "",
    "பொழுதில்": "",
    "சிரித்தான்": "சிரிச்சான்",
    "கவர்ந்தது": "ஈர்த்தது",
    "நாவிலோர்": "பேசுறது",
    "இனிமை": "இனிமையா",
    "சிந்தினாள்": "இருந்தது",
    "பனிமூட்டம்": "பனியால்",
    "மூடியது": "மூடப்பட்டுருச்சு",
    "தந்தை": "அப்பா",
    "பிள்ளைகளுக்கு": "பிள்ளைகளுக்கு",
    "வழி": "வழிகாட்ட",
    "காட்டினான்": "காட்டினான்",
    "புன்னகை": "சிரிப்பு",
    "கொள்ளை": "கவர்ந்த",
    "கொண்டது": "கவர்ந்தது",
    "முயற்சியால்": "கஷ்டப்பட்டதால",
    "அடைந்தான்": "அடைந்தான்",
    "பாசம்": "அன்பு",
    "அளவில்லாதது": "அளவில்லாதது",
    "இசைபோல்": "இசையா",
    "பறந்தன": "பறந்தன",
    "வாசனை": "வாசனை",
    "நெஞ்சை": "மனசை",
    "நெகிழச்செய்யும்": "தூண்டும்",
    "இலக்கை": "டார்க்கெட்",
    "நோக்கி": "நோக்கி",
    "பயணித்தான்": "போனான்",
    "நினைவுகள்": "நினைவுகள்",
    "உள்ளத்தை": "மனசை",
    "உருக்கும்": "உருக்குது",
    "வார்த்தைகள்": "வார்த்தைகள்",
    "வாழ்வை": "வாழ்க்கையை",
    "மாற்றின": "மாற்றியது"
}

# --- 3. Enhanced sentence mappings (your original rules) ---
REWRITE_RULES = {
    "அவன் பள்ளிக்குச் செல்கிறான் மற்றும் பாடுகின்றான்": "அவன் பள்ளிக்குப் போயி பாடுகிறான்",
    "அவள் மிகவும் அழகாக விழாவிற்கு வந்தாள்": "அவள் ரொம்ப அழகா விழாவுக்கு வந்தது",
    "அவர் மக்கள் முன் உரையாற்றுகின்றார் மற்றும் யாதர்த்தம் பற்றி பேசுகின்றார்": "அவர் பேசுறார் மற்றும் உண்மை பற்றி சொல்றார்",
    "நான் என் வாழ்நாளில் மேற்கொண்ட செயல்களைப்பற்றி சிந்திக்கிறேன்": "நான் என் வாழ்க்கையில் செய்த செயல்களைப் பற்றி யோசிக்கிறேன்",
    "அவன் இலக்கியம் மீது பெரும் ஆர்வம் கொண்டவன் மற்றும் தினமும் படிக்கிறான்": "அவன் இலக்கியத்தை ரொம்ப விருப்பம் பண்ணுவான், தினமும் படிக்கிறான்",
    "வாடையும் பிரிந்தினோர்க்கு அழலே": "பிரிந்தவர்களுக்கு வாடைக் காற்றும் நெருப்புபோல இருக்கு",
    "அவளது குரல் மலரென மென்மையாய் இருந்தது": "அவள் குரல் ரொம்ப மென்மையா இருந்தது",
    "மழை பெய்யும் சுழற்காற்றின் இசை இனிமையானது": "மழைக்காற்று வரும் சத்தம் ரொம்ப இனிமையா இருக்கு",
    "சிறகு விரித்து வானில் பறந்தான் பறவை": "பறவை வானத்தில் சிறகுபோட்டு பறந்தது",
    "மண்ணின் மணம் ஆனந்தம் தரும்": "மண்ணோட வாசனை சந்தோஷம் தருது",
    "அவள் இளமை பெண்ணாக மழையில் நடக்கிறாள்": "அவள் ஒரு யுவதி மாதிரி மழையில நடக்கிறா",
    "அந்த இடம் பசுமை நிறைந்தது மற்றும் அமைதியாக இருந்தது": "அந்த இடம் ரொம்ப பசுமையா அமைதியா இருந்தது",
    "அவன் பசிக்கொண்டு இருந்தபோதும் சாப்பிடவில்லை": "அவன் பசிக்குமே சாப்பிடல",
    "நதி ஓசை காற்றுடன் இசைவதாக இருந்தது": "நதியின் சத்தம் காற்றோட சேர்ந்து இனிமையா இருந்தது",
    "அவள் மகிழ்ச்சியுடன் பூக்கள் சேகரித்தாள்": "அவள் சந்தோஷமா பூக்கள் எடுத்தா",
    "அவன் அறிவுடன் பேசினான் மற்றும் மக்களை பிரமிக்கச் செய்தான்": "அவன் புத்திசாலியா பேசினான், எல்லாரையும் ஆச்சரியப்படுத்தினான்",
    "அவள் தன் தாய் மீது மிகுந்த பக்தி கொண்டவள்": "அவள் தன்னோட அம்மாவை ரொம்ப மதிக்கறா",
    "காற்றின் ஸ்பரிசம் என் மனதை தென்றல் போல தென்றலாக்கியது": "காற்றோட தொடுதல் மனசுக்கே நிம்மதியா இருந்தது",
    "அவன் என் மனதில் இடம் பிடித்தான்": "அவன் என் மனசுல இடம் கட்டிக்கிட்டான்",
    "அவன் கனவு காணும் பொழுதில் சிரித்தான்": "அவன் கனவுல சிரிச்சான்",
    "அந்த காட்சி என் கண்களை கவர்ந்தது": "அந்த காட்சி என் கண்களை ஈர்த்தது",
    "அவள் நாவிலோர் இனிமை சிந்தினாள்": "அவள் பேசுறது ரொம்ப இனிமையா இருந்தது",
    "பனிமூட்டம் காடுகள் அனைத்தையும் மூடியது": "பனியால் காடு முழுக்க மூடப்பட்டுருச்சு",
    "அந்த தந்தை தன் பிள்ளைகளுக்கு நல்ல வழி காட்டினான்": "அந்த அப்பா பிள்ளைகளுக்கு நல்லா வழிகாட்டினான்",
    "அவள் புன்னகை என் மனதை கொள்ளை கொண்டது": "அவளோட சிரிப்பு என் மனசை கவர்ந்தது",
    "அவன் முயற்சியால் வெற்றியை அடைந்தான்": "அவன் கஷ்டப்பட்டதால வெற்றி அடைந்தான்",
    "அவள் தாய் மீது கொண்ட பாசம் அளவில்லாதது": "அவளோட அம்மாவுக்கான அன்பு அளவில்லாதது",
    "பறவைகள் வானில் இசைபோல் பறந்தன": "பறவைகள் வானத்தில் இசையா பறந்தன",
    "மழையின் வாசனை என் நெஞ்சை நெகிழச்செய்யும்": "மழை வாசனை என் மனசை தூண்டும்",
    "அவன் தன் இலக்கை நோக்கி பயணித்தான்": "அவன் தன் டார்க்கெட் நோக்கி போனான்",
    "அந்த நினைவுகள் என் உள்ளத்தை உருக்கும்": "அந்த நினைவுகள் என் மனசை உருக்குது",
    "அவள் வார்த்தைகள் என் வாழ்வை மாற்றின": "அவளோட வார்த்தைகள் என் வாழ்க்கையை மாற்றியது",
    "படகு அமைதியாக நதியில் சறுக்கியது": "படகு நதில சும்மா சறுக்கிச்சு போனது",
    "அவன் தன் நண்பனை நம்பிக்கையுடன் பார்த்தான்": "அவன் தன் தோழனை நம்பிகையோட பார்த்தான்",
    "பசுமை மலைகள் மனதுக்கு அமைதியை தருகின்றன": "பசுமையான மலைகள் மனசுக்கு நிம்மதியா இருக்கு",
    "அவள் கண்ணின் ஒளி சந்தோஷத்தை சொல்கிறது": "அவளோட கண் பிரகாசம் சந்தோஷத்தைக் காட்டுது",
    "அவன் இசையில் மூழ்கி இருந்தான்": "அவன் பாடலோட முழுசா ஆழ்ந்துருந்தான்",
    "அந்த நகரம் தொன்மையைக் கொண்டது": "அந்த ஊர் பழைய வரலாற்று கதை சொல்றது",
    "அவள் குரலில் வேதனையும் அழகும் கலந்து இருந்தன": "அவளோட குரல் கலக்கல – கவலையும் அழகும் இருக்கு",
    "காற்றில் பறக்கும் துள்ளல் மரபைக் கூறுகிறது": "காற்றோட ஆட்டம் பழக்கத்தையும் காட்டுது",
    "அவன் போராட்டத்தில் உறுதியுடன் இருந்தான்": "அவன் போராட்டத்தில் பிடிவாதமா இருந்தான்",
    "அவள் மழையை ரசித்தாள்": "அவள் மழையைய ரசிச்சா",
    "முழு நிலவு இரவில் பிரகாசிக்கின்றது": "முழு நிலா இரவில பிரகாசிச்சுது",
    "அந்த வண்ணங்கள் விழிகளை கவருகின்றன": "அந்த கலருகள் பார்வையை ஈர்க்குது",
    "அவன் நண்பனுக்காக தன்னையே விட்டான்": "அவன் தோழனுக்காக தன்னையே தியாகம் பண்ணினான்",
    "அவள் சொற்கள் என் உள்ளத்தில் ஒலிக்கின்றன": "அவளோட வார்த்தைகள் என் மனசுல ஒலிக்குது",
    "அந்த கதை மனதைக் குழப்பியது": "அந்த கதை கலக்கு செய்தது",
    "பூமி பசுமையாக இருக்க வேண்டும்": "நம்ம பூமி பசுமையாக இருக்கணும்",
    "அவன் பிறருக்கு உதவ விரும்புகிறான்": "அவன் எல்லாருக்குமே உதவ ஆசைப்படுறான்",
    "மழைதானின் சத்தம் அமைதியாக இருந்தது": "மழை சத்தம் ரொம்ப சும்மா இருந்தது",
    "அவள் எழுத்துக்கள் உள்ளத்தை தொடுகின்றன": "அவளோட எழுத்துகள் மனசை தொட்டது",
    "அந்த சந்திப்பு என் வாழ்க்கையின் திருப்புமுனை": "அந்த சந்திப்பு என் வாழ்க்கையை மாற்றியது",
    "அவன் பூரண அன்புடன் வாழ்கிறான்": "அவன் முழு மனசோட வாழ்றான்",
    "அந்த இடம் பழமையை பிரதிபலிக்கின்றது": "அந்த இடம் பழைய காலத்தை நினைவு செய்றது",
    "அவள் தன் கனவுகளை பின்தொடர்கிறாள்": "அவள் தன் கனவுகளுக்கு பின் போறா",
    "அவன் ஒளியாய் என் வாழ்க்கையில் வந்தான்": "அவன் வெளிச்சமா என் வாழ்க்கைக்கு வந்தான்",
    "அந்த நேரம் மறக்க முடியாதது": "அந்த நேரம் மறக்கவே முடியல",
    "அவள் எல்லோரையும் அன்புடன் வரவேற்றாள்": "அவள் எல்லாரையும் அன்பா வரவேச்சா",
    "அவன் காலத்துக்கு முன்னே சென்று சிந்திக்கிறான்": "அவன் அடிக்கடி முன் நெனச்சு பேசுவான்",
    "அந்த இடத்தில் அமைதி பரவியது": "அந்த இடம் ரொம்ப அமைதியா இருந்தது",
    "அவள் ஒளியாய் பசுமையை பரப்பினாள்": "அவள் வெளிச்சமா பசுமை காட்டினா",
    "அந்த நினைவுகள் என் கனவுகளை அலங்கரிக்கின்றன": "அந்த ஞாபகங்கள் என் கனவுகளை அழகாக்குது",
    "அவன் தன் எண்ணங்களை எழுத்தில் பதிவு செய்தான்": "அவன் தன் சிந்தனைகளை எழுதி வச்சான்",
    "அந்த வார்த்தைகள் எனக்கு உந்துதலை அளித்தன": "அந்த வார்த்தைகள் எனக்கு தூண்டலா இருந்தது",
    "அவள் ஒரு கவிதை போல நடந்தாள்": "அவள் ஒரு கவிதை மாதிரி நடந்தா",
    "அந்த மேகம் கனத்ததாக இருந்தது": "அந்த மேகம் ரொம்ப கருப்பா இருந்தது",
    "அவன் தன் பாட்டை பாடிக்கொண்டே சென்றான்": "அவன் தன் பாட்டை பாட்டிக்கிட்டே போனான்",
    "அந்த வேளையில் காற்று துள்ளலாக இருந்தது": "அந்த நேரத்துல காற்று ஜோர் காற்றா இருந்தது",
    "அந்த வானம் நட்சத்திரங்களால் பளிச்சிடுகிறது": "வானம் நட்சத்திரங்களால பிரகாசிச்சுது",
    "அவள் நடையின் மென்மை என் கண்களை கவர்ந்தது": "அவளோட நடைய மென்மையா இருந்தது, என் கண் கவர்ந்தது",
    "அவன் புத்தகத்தை ஆசையோடு படித்தான்": "அவன் புத்தகத்தை விருப்பமா படிச்சான்",
    "மழை துளிகள் கண்ணை மகிழ்வித்தன": "மழைத்துளிகள் பார்வைக்கு சந்தோஷம் தந்தது",
    "அந்த இசை என் உள்ளத்தில் ஒலிக்கிறது": "அந்த பாட்டு என் மனசுல ஒலிக்குது",
    "அவன் வாழ்க்கையின் நிழலாக இருந்து வந்தான்": "அவன் வாழ்க்கையில ஒரு நிழலா இருந்தான்",
    "அவளது வார்த்தைகள் தேன் போல இனிமையானவை": "அவளோட வார்த்தைகள் தேன்லாம் கூட இனிமையா இருக்கும்",
    "அந்த ஓவியம் என் எண்ணங்களை மாற்றியது": "அந்த ஓவியம் என் யோசனையை மாற்றிச்சு",
    "அவன் மனதை கண்ணீரில் வெளிப்படுத்தினான்": "அவன் கண்ணீரோட தான் உணர்ச்சிகளை காட்டினான்",
    "அவள் எழுதிய கவிதை உயிருடன் இருந்தது": "அவளோட கவிதை உயிரோட இருந்தது",
    "அந்த அனுபவம் எனக்கு வாழ்கையை கற்றுத்தந்தது": "அந்த அனுபவம் எனக்கு வாழ்க்கை பாடம் கற்றுத்தந்தது",
    "அவன் விடாமுயற்சியால் முன்னேறினான்": "அவன் போராடினதால முன்னேறினான்",
    "அந்த காகிதம் நினைவுகளை தூண்டியது": "அந்த பேப்பர் பழைய ஞாபகங்களை நினைவுபடுத்திச்சு",
    "அவளது முகத்தில் ஒளி வீசியது": "அவள முகம் பிரகாசிச்சுச்சு",
    "அவன் தனிமையில் தன்னை மறந்து வாழ்ந்தான்": "அவன் தனிமையிலேயே தன்னை மறந்துட்டான்",
    "அந்த கடல் அலைகள் சுருதியில் பாடின": "கடல் அலைகள் ஒரு இசையா ஒலிச்சது",
    "அவள் ஒரு ஓவியத்திலிருந்து வெளிவந்தவள்போல் தெரிந்தாள்": "அவள பாத்ததும் ஓவியம் மாதிரி இருந்தா",
    "அவன் இளமையில் அறிவுடன் பவனி செய்தான்": "அவன் இளமையிலேயே புத்திசாலியா இருந்தான்",
    "மழை என் இதயத்தில் துளிர் விட்டது": "மழை என் மனசில ஒரு சந்தோஷத்தை கிளப்புச்சு",
    "அந்த சொற்கள் என் மனதை உருக்கியது": "அந்த வார்த்தைகள் என் மனசை உருக்கிச்சு",
    "அவள் பொன்னே போன்ற அழகுடன் வந்தாள்": "அவள் பொன்னு மாதிரி அழகா வந்தா",
    "அவன் சொந்தங்களை நேசிப்பவன்": "அவன் குடும்பத்தை ரொம்ப அன்பா பாக்கறான்",
    "அந்த இடம் பழமையை பேசுகிறது": "அந்த இடம் பழைய காலங்களை நினைவுபடுத்துது",
    "அவளது பார்வையில் அன்பும் ஆசையும் காணப்பட்டது": "அவளோட பார்வையில அன்பும் ஆசையும் தெரிந்தது",
    "அந்த நட்சத்திரம் இருளில் ஒளிர்ந்தது": "அந்த நட்சத்திரம் இருள்ல ஒளிச்சது",
    "அவன் அன்புடன் பேசினான் மற்றும் அனைவரையும் கவர்ந்தான்": "அவன் மென்மையா பேசினான், எல்லாரையும் கவர்ந்தான்",
    "அந்த காடுகள் இயற்கையின் விருந்தினர் போல இருந்தன": "அந்த காடுகள் இயற்கையோட நெருக்கமான இடமா இருந்தது",
    "அவள் நெஞ்சத்தில் உணர்ச்சி வெள்ளமாக ஒழுகியது": "அவள மனசில உணர்ச்சி கரை வெச்சு வந்துச்சு",
    "அந்த நதி என் எண்ணங்களை வாரி எடுத்தது": "அந்த நதி என் யோசனைகளை எடுத்து போச்சு",
    "அவன் வெற்றியை எண்ணி முகம் மலர்ந்தது": "அவன் வெற்றி நினைச்சு சந்தோஷமா சிரிச்சான்"

}


# --- 4. Preload modern Tamil sentences for fallback search ---
MODERN_SENTENCE_BANK = list(REWRITE_RULES.values())

# --- 5. Enhanced word-level transformation ---
def modernize_words(text):
    """Apply word-level transformations"""
    words = text.split()
    modernized_words = []
    
    for word in words:
        # Check for exact match
        if word in WORD_MAPPINGS:
            modernized_words.append(WORD_MAPPINGS[word])
        # Check for partial matches (suffixes/prefixes)
        else:
            found_replacement = False
            for classical_word, modern_word in WORD_MAPPINGS.items():
                if classical_word in word and len(classical_word) > 3:  # Avoid short matches
                    replaced_word = word.replace(classical_word, modern_word)
                    modernized_words.append(replaced_word)
                    found_replacement = True
                    break
            
            if not found_replacement:
                modernized_words.append(word)
    
    return ' '.join(modernized_words)

# --- 6. Similarity scoring function ---
def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    return SequenceMatcher(None, text1, text2).ratio()

# --- 7. Enhanced semantic search ---
def find_best_semantic_match(text, candidates, threshold=0.3):
    """Find the best semantic match from candidates"""
    try:
        # Use inltk's get_similar_sentences
        similar_sentences = get_similar_sentences(text, 'ta', min(5, len(candidates)), candidates)
        
        if similar_sentences:
            # Calculate similarity scores for the top matches
            best_match = None
            best_score = 0
            
            for candidate in similar_sentences:
                score = calculate_similarity(normalize(text), normalize(candidate))
                if score > best_score and score > threshold:
                    best_match = candidate
                    best_score = score
            
            return best_match, best_score
    except Exception as e:
        print(f"Semantic search failed: {e}")
    
    return None, 0

# --- 8. Pattern-based transformations ---
def apply_common_patterns(text):
    """Apply common classical to modern Tamil patterns"""
    patterns = [
        # Verb endings
        (r'கின்றான்$', 'கிறான்'),
        (r'கின்றார்$', 'கிறார்'),
        (r'கின்றாள்$', 'கிறா'),
        (r'கின்றன$', 'கின்றன'),
        (r'கின்றது$', 'குது'),
        
        # Common suffixes
        (r'இற்கு$', 'உக்கு'),
        (r'ஆய்$', 'ஆ'),
        (r'உடன்$', 'ஓட'),
        (r'ஆல்$', 'ஆல'),
        
        # Common word transformations
        (r'\bமிகவும்\b', 'ரொம்ப'),
        (r'\bஅமைதியாக\b', 'அமைதியா'),
        (r'\bஅழகாக\b', 'அழகா'),
        (r'\bமென்மையாய்\b', 'மென்மையா'),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
    return result

# --- 9. Main enhanced function ---
def modernize_text(text, lang='ta', use_word_level=True, use_patterns=True, similarity_threshold=0.3):
    """
    Enhanced function to modernize classical Tamil text
    
    Args:
        text: Input classical Tamil text
        lang: Language code (default: 'ta')
        use_word_level: Whether to apply word-level transformations
        use_patterns: Whether to apply pattern-based transformations
        similarity_threshold: Minimum similarity score for semantic matching
    
    Returns:
        tuple: (modernized_text, english_translation, matched_rule, confidence_score)
    """
    cleaned_text = normalize(text)
    modified = cleaned_text
    matched_rule = None
    confidence_score = 0.0
    method_used = "original"
    
    # Step 1: Try exact rule-based match
    for classical, modern in REWRITE_RULES.items():
        if normalize(classical) == cleaned_text:
            modified = modern
            matched_rule = classical
            confidence_score = 1.0
            method_used = "exact_rule"
            break
    
    # Step 2: If no exact match, try word-level transformations
    if modified == cleaned_text and use_word_level:
        word_modernized = modernize_words(cleaned_text)
        if word_modernized != cleaned_text:
            modified = word_modernized
            confidence_score = 0.8
            method_used = "word_level"
    
    # Step 3: Apply pattern-based transformations
    if use_patterns:
        pattern_modernized = apply_common_patterns(modified)
        if pattern_modernized != modified:
            modified = pattern_modernized
            if method_used == "original":
                confidence_score = 0.6
                method_used = "pattern_based"
    
    # Step 4: Semantic fallback using enhanced search
    if modified == cleaned_text or confidence_score < similarity_threshold:
        try:
            # First try with rule values
            best_match, score = find_best_semantic_match(cleaned_text, MODERN_SENTENCE_BANK, similarity_threshold)
            
            if best_match and score > confidence_score:
                modified = best_match
                confidence_score = score
                method_used = "semantic_rules"
            
            # If still not found, try with a broader candidate set
            if confidence_score < similarity_threshold:
                # You can expand this with more modern Tamil sentences
                extended_candidates = MODERN_SENTENCE_BANK + [
                    "அவன் போறான்", "அவள் வரா", "நான் செய்றேன்", "அது இருக்கு",
                    "இது நல்லா இருக்கு", "அவங்க பேசுறாங்க", "நீ என்ன செய்றே",
                    "எங்க போறீங்க", "எப்ப வரீங்க", "எதுக்கு அழுறே"
                ]
                
                best_match, score = find_best_semantic_match(cleaned_text, extended_candidates, similarity_threshold)
                if best_match and score > confidence_score:
                    modified = best_match
                    confidence_score = score
                    method_used = "semantic_extended"
                    
        except Exception as e:
            print(f"Enhanced semantic search failed: {e}")
    
    # Step 5: Translate to English
    try:
        english_translation = GoogleTranslator(source='ta', target='en').translate(modified)
    except Exception as e:
        english_translation = f"(Translation failed: {e})"
    
    return {
        'modernized': modified,
        'english': english_translation,
        'matched_rule': matched_rule,
        'confidence': confidence_score,
        'method': method_used,
        'original': text
    }

# --- 10. Batch processing function ---
def modernize_batch(texts, **kwargs):
    """Process multiple texts at once"""
    results = []
    for text in texts:
        result = modernize_text(text, **kwargs)
        results.append(result)
    return results

# --- 11. Example usage and testing ---
if __name__ == "__main__":
    # Test cases including sentences not in the original rules
    test_sentences = [
        "அவன் பள்ளிக்குச் செல்கிறான் மற்றும் பாடுகின்றான்",  # In rules
        "அவள் மிகவும் அழகாக நடக்கிறாள்",  # Partial in rules
        "நான் புத்தகம் படிக்கின்றேன்",  # Not in rules
        "அவர் மக்களுக்கு உதவுகின்றார்",  # Not in rules
        "குழந்தைகள் விளையாடுகின்றனர்",  # Not in rules
        "மழை பெய்கின்றது மற்றும் காற்று வீசுகின்றது"  # Not in rules
    ]
    
    print("=== Enhanced Classical to Modern Tamil Conversion ===\n")
    
    for sentence in test_sentences:
        result = modernize_text(sentence)
        print(f"Original: {result['original']}")
        print(f"Modernized: {result['modernized']}")
        print(f"English: {result['english']}")
        print(f"Method: {result['method']}")
        print(f"Confidence: {result['confidence']:.2f}")
        if result['matched_rule']:
            print(f"Matched Rule: {result['matched_rule']}")
        print("-" * 50)