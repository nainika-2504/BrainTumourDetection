
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(2048, 4)
    )
    model.load_state_dict(torch.load(
        "best_brain_model_v2.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model             = load_model()
feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
feature_extractor.eval()
mean_features     = np.load("ood_brain_mean.npy")
cov_inv           = np.load("ood_brain_cov_inv.npy")
ood_threshold     = np.load("ood_brain_threshold.npy")[0]
class_names       = ["glioma_tumor", "meningioma_tumor",
                      "no_tumor", "pituitary_tumor"]
tumor_names       = {
    "glioma_tumor"    : "Glioma Tumor",
    "meningioma_tumor": "Meningioma Tumor",
    "no_tumor"        : "No Tumor",
    "pituitary_tumor" : "Pituitary Tumor"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def mahalanobis_distance(tensor):
    with torch.no_grad():
        feat = feature_extractor(tensor)
        feat = feat.squeeze(-1).squeeze(-1).cpu().numpy()[0]
    diff = feat - mean_features
    return float(np.sqrt(diff @ cov_inv @ diff))

def get_severity(prediction, confidence):
    if prediction == "no_tumor":
        return "NONE", "No tumor detected"
    if confidence >= 95:
        return "HIGH",             "Urgent medical attention required"
    elif confidence >= 80:
        return "MODERATE",         "Medical consultation recommended soon"
    else:
        return "LOW CONFIDENCE",   "Further imaging recommended"

def get_location(grayscale_cam):
    h, w   = grayscale_cam.shape
    top    = grayscale_cam[:h//3, :].mean()
    middle = grayscale_cam[h//3:2*h//3, :].mean()
    bottom = grayscale_cam[2*h//3:, :].mean()
    left   = grayscale_cam[:, :w//2].mean()
    right  = grayscale_cam[:, w//2:].mean()
    vert   = max({"Frontal region": top, "Central region": middle,
                  "Posterior region": bottom}, key=lambda k: {
                  "Frontal region": top, "Central region": middle,
                  "Posterior region": bottom}[k])
    horiz  = "Left hemisphere" if left > right else "Right hemisphere"
    return f"{vert}, {horiz}"

def get_recommendations(prediction, severity):
    if prediction == "no_tumor":
        return ["No tumor detected",
                "Continue regular health checkups",
                "Consult doctor if symptoms persist",
                "Follow-up MRI in 6-12 months if recommended"]
    base = ["DISCLAIMER: AI suggestion only - consult a neurologist immediately"]
    recs = {
        "glioma_tumor": {
            "HIGH":           ["Seek immediate neurosurgical consultation",
                               "Biopsy required to determine grade (I-IV)",
                               "Treatment may include surgery, radiation, chemotherapy",
                               "Genetic testing (IDH, MGMT) may guide treatment"],
            "MODERATE":       ["Schedule neurology appointment within 1 week",
                               "MRI with contrast recommended",
                               "Biopsy may be required"],
            "LOW CONFIDENCE": ["Additional imaging needed",
                               "Consult neurologist for further evaluation"]
        },
        "meningioma_tumor": {
            "HIGH":           ["Neurosurgical consultation recommended",
                               "Surgery may be required depending on size/location",
                               "Regular monitoring with serial MRIs"],
            "MODERATE":       ["Schedule neurology appointment within 2 weeks",
                               "Watch and wait approach may be suitable",
                               "Regular MRI monitoring every 6 months"],
            "LOW CONFIDENCE": ["Follow-up MRI with contrast recommended",
                               "Consult neurologist for evaluation"]
        },
        "pituitary_tumor": {
            "HIGH":           ["Endocrinology and neurosurgery consultation urgently",
                               "Hormone level blood tests required immediately",
                               "Visual field testing recommended",
                               "Transsphenoidal surgery may be required"],
            "MODERATE":       ["Schedule endocrinologist appointment",
                               "Hormone panel blood work recommended",
                               "Monitor vision changes carefully"],
            "LOW CONFIDENCE": ["Dedicated pituitary MRI recommended",
                               "Hormone levels should be checked"]
        }
    }
    return base + recs.get(prediction, {}).get(severity,
           ["Consult a neurologist immediately"])

def predict(image, patient_name, patient_age):
    if image is None:
        return "Please upload an image", None, None, None
    if not patient_name:
        patient_name = "Unknown"
    if not patient_age:
        patient_age  = "N/A"

    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    dist = mahalanobis_distance(tensor)
    if dist > ood_threshold:
        return ("REJECTED - Please upload a T1-weighted brain MRI scan only.\n"
                "CT scans, X-rays and other imaging types are not supported."),                {"Not a brain MRI": 1.0}, None, None

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        pred    = outputs.argmax(1).item()
        conf    = probs[0][pred].item() * 100

    prediction = class_names[pred]

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cam           = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=tensor,
                        targets=[ClassifierOutputTarget(pred)])[0]
    rgb_img = np.array(img.resize((224,224))).astype(np.float32) / 255.0
    gradcam_viz = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    severity, _ = get_severity(prediction, conf)
    location    = get_location(grayscale_cam)
    recs        = get_recommendations(prediction, severity)

    orig_path    = "/tmp/brain_orig.png"
    gradcam_path = "/tmp/brain_gradcam.png"
    img.resize((224,224)).save(orig_path)
    Image.fromarray(gradcam_viz).save(gradcam_path)

    pdf_path = f"/tmp/brain_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    doc      = SimpleDocTemplate(pdf_path, pagesize=A4,
                                  rightMargin=40, leftMargin=40,
                                  topMargin=40,   bottomMargin=40)
    story    = []
    t_style  = ParagraphStyle("t", fontSize=20, fontName="Helvetica-Bold",
                               textColor=colors.HexColor("#1a237e"), spaceAfter=5)
    s_style  = ParagraphStyle("s", fontSize=11, fontName="Helvetica",
                               textColor=colors.grey, spaceAfter=15)
    se_style = ParagraphStyle("se", fontSize=13, fontName="Helvetica-Bold",
                               textColor=colors.HexColor("#1a237e"), spaceAfter=8)
    n_style  = ParagraphStyle("n", fontSize=10, fontName="Helvetica", spaceAfter=4)
    d_style  = ParagraphStyle("d", fontSize=8,  fontName="Helvetica-Oblique",
                               textColor=colors.red, spaceAfter=4)

    story.append(Paragraph("Brain Tumour Detection Report", t_style))
    story.append(Paragraph("AI-Assisted Neurological Imaging Analysis", s_style))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Patient Information", se_style))
    pt = Table([["Patient Name", patient_name],
                ["Age",          f"{patient_age} years"],
                ["Scan Type",    "T1-Weighted Brain MRI"],
                ["Report Date",  datetime.now().strftime("%Y-%m-%d %H:%M")],
                ["Report ID",    f"BT-{datetime.now().strftime('%Y%m%d%H%M%S')}"]],
               colWidths=[2*inch, 4*inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8eaf6")),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    story.append(pt)
    story.append(Spacer(1, 15))
    story.append(Paragraph("Analysis Result", se_style))
    result_color = colors.green if prediction == "no_tumor" else colors.red
    rt = Table([["Diagnosis",  tumor_names.get(prediction, prediction)],
                ["Confidence", f"{conf:.2f}%"],
                ["Severity",   severity],
                ["Location",   location if prediction != "no_tumor" else "N/A"]],
               colWidths=[2*inch, 4*inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8eaf6")),
        ("BACKGROUND", (1,0), (1,0),  result_color),
        ("TEXTCOLOR",  (1,0), (1,0),  colors.white),
        ("FONTNAME",   (0,0), (-1,-1),"Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    story.append(rt)
    story.append(Spacer(1, 15))
    story.append(Paragraph("MRI Analysis", se_style))
    it = Table([[RLImage(orig_path,    width=2.5*inch, height=2.5*inch),
                 RLImage(gradcam_path, width=2.5*inch, height=2.5*inch)],
                [Paragraph("Original MRI", n_style),
                 Paragraph("Grad-CAM Heatmap", n_style)]],
               colWidths=[3*inch, 3*inch])
    it.setStyle(TableStyle([("ALIGN",   (0,0), (-1,-1), "CENTER"),
                             ("PADDING", (0,0), (-1,-1), 10),
                             ("BOX",     (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(it)
    story.append(Spacer(1, 15))
    story.append(Paragraph("Clinical Recommendations", se_style))
    for i, rec in enumerate(recs):
        if i == 0 and prediction != "no_tumor":
            story.append(Paragraph(f"{rec}", d_style))
        else:
            story.append(Paragraph(f"• {rec}", n_style))
    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model and is NOT a substitute "
        "for professional medical advice. Always consult a qualified neurologist.",
        d_style))
    doc.build(story)

    result = (f"{tumor_names.get(prediction, prediction).upper()}\n"
              f"Confidence: {conf:.2f}%\n"
              f"Severity: {severity}\n"
              f"Location: {location}")
    confs  = {tumor_names[c]: round(probs[0][i].item(), 3)
               for i, c in enumerate(class_names)}
    return result, confs, gradcam_viz, pdf_path

with gr.Blocks(title="Brain Tumour Detection") as demo:
    gr.Markdown("""
    # Brain Tumour Detection
    ### AI-Assisted Neurological Imaging Analysis
    > Upload a T1-weighted brain MRI to get instant analysis + downloadable PDF report
    ---
    """)
    with gr.Row():
        with gr.Column():
            image_input  = gr.Image(label="Upload Brain MRI")
            patient_name = gr.Textbox(label="Patient Name",
                                       placeholder="Enter patient name")
            patient_age  = gr.Textbox(label="Patient Age",
                                       placeholder="Enter patient age")
            submit_btn   = gr.Button("Analyze & Generate Report",
                                      variant="primary")
        with gr.Column():
            result_text  = gr.Textbox(label="Analysis Result", lines=5)
            confidence   = gr.Label(label="Confidence Scores")
            gradcam_out  = gr.Image(label="Grad-CAM Heatmap")
            pdf_output   = gr.File(label="Download PDF Report")
    submit_btn.click(fn=predict,
                     inputs=[image_input, patient_name, patient_age],
                     outputs=[result_text, confidence, gradcam_out, pdf_output])
    gr.Markdown("""
    ---
    **Model**: ResNet50 | **Accuracy**: 85.53% (with TTA) | **OOD Detection**: Enabled
    > For educational purposes only. Not a substitute for medical advice.
    """)
demo.launch()
