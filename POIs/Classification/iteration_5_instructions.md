# הוראות לאיטרציה 5 - Target Population Classification

## מה הקוד עושה?

הקוד שנוסף למחברת (התא האחרון) מבצע את התהליך הבא:

### שלב 1: טעינת נתונים מתויגים
- טוען את כל הנתונים המתויגים מאיטרציות 1-4:
  - `iteration_1_locals_vs_diaspora_translated.csv`
  - `iteration_2_manual_labels_target_population.csv`
  - `iteration_3_manual_labels_target_population.csv`
  - `iteration_4_manual_labels_target_population.csv` (קובץ חדש שתצטרך ליצור)

### שלב 2: אימון מודל
- מאמן מודל Binary Logistic Regression על מאפיינים משולבים:
  - TF-IDF של טקסט (username + display_name + description)
  - מאפיינים מהונדסים:
    - זיהוי כתב ערבי
    - זיהוי אותיות פרסיות ספציפיות (پ چ ژ گ)
    - זיהוי וריאציות פרסיות
    - קיום location
    - log של מספר followers

### שלב 3: חיזוי על כל המועמדים
- מחזה הסתברויות עבור כל המשתמשים
- מחשב ציוני uncertainty

### שלב 4: יצירת קבצים
הקוד יוצר 2 קבצים:

1. **`iteration_4_unlabeled_users_predictions_target_population.csv`**
   - כולל את כל המועמדים עם ציוני הוודאות וההסתברויות
   - עמודות עיקריות:
     - `prob_0`, `prob_1`: הסתברויות לכל מחלקה
     - `confidence_level`: רמת וודאות
     - `uncertainty_score`: ציון אי-וודאות (גבוה = לא בטוח)
     - `predicted_class`: חיזוי בינארי (0/1)
     - `predicted_class_with_unknown`: חיזוי עם קטגוריית "לא ידוע" (0/1/2)

2. **`iteration_5_manual_labeling_candidates_target_population.csv`**
   - **זהו הקובץ שעליך לתייג ידנית!**
   - מכיל 100 המשתמשים עם הציון uncertainty הגבוה ביותר
   - אלו המקרים שהמודל הכי לא בטוח לגביהם

---

## מה עליך לעשות?

### שלב 1: הרצת הקוד
הרץ את התא האחרון במחברת. הקוד יפיק את 2 הקבצים הנ"ל בתיקיה:
```
POIs/Classification/
```

### שלב 2: תיוג ידני
פתח את הקובץ:
```
POIs/Classification/iteration_5_manual_labeling_candidates_target_population.csv
```

הוסף עמודה בשם **`manual_target_population`** ותייג כל משתמש:

- **0** = Not target population (לא אוכלוסיית המטרה)
- **1** = Target population (אוכלוסיית המטרה - איראנים חיים באיראן)
- **2** = Unknown (אם אתה לא בטוח)

**טיפים לתיוג:**
- התמקד במשתמשים עם `uncertainty_score` גבוה
- שים לב ל-`description_en`, `location`, `username`
- זכור: target population = איראנים שחיים באיראן (לא דיאספורה)

### שלב 3: שמור את הקובץ המתויג
שמור את הקובץ המתויג בשם:
```
POIs/Classification/iteration_4_manual_labels_target_population.csv
```

**חשוב:** שמור בקידוד UTF-8 וודא שהעמודה `manual_target_population` קיימת.

---

## איטרציה הבאה (6)

לאחר שתסיים את התיוג:
1. הרץ את הקוד שוב (יתעדכן אוטומטית לאיטרציה 6)
2. או העתק את הקוד ושנה:
   - הוסף את `iteration_4_manual_labels_target_population.csv` לרשימת הקבצים
   - שנה את שמות קבצי הפלט ל-`iteration_5_*` ו-`iteration_6_*`

---

## סיכום קבצים

| שם קובץ | מטרה | פעולה נדרשת |
|---------|------|-------------|
| `iteration_5_manual_labeling_candidates_target_population.csv` | TOP 100 משתמשים לא בטוחים | **תייג ידנית** |
| `iteration_4_unlabeled_users_predictions_target_population.csv` | כל החיזויים | לעיון בלבד |
| `iteration_4_manual_labels_target_population.csv` | התוויות שלך לאחר תיוג | **שמור כאן** |

---

## שאלות נפוצות

**ש: למה דווקא 100 משתמשים?**
ת: זהו מספר מאוזן - מספיק כדי לשפר את המודל, לא יותר מדי שיהיה מעייף לתייג.

**ש: מה אם אני לא בטוח במשתמש מסוים?**
ת: תייג אותו כ-2 (unknown). המודל ידלג עליו באימון.

**ש: האם אני צריך לתייג את כל 100?**
ת: רצוי, אבל אם תתייג פחות זה עדיין יעזור לשפר את המודל.

**ש: מתי לעצור את האיטרציות?**
ת: כאשר רוב המשתמשים ב-TOP 100 הם באמת מקרי ספר או כאשר הדיוק של המודל מספיק טוב.
