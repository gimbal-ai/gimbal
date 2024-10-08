# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/render_data.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.util import color_pb2 as mediapipe_dot_util_dot_color__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n mediapipe/util/render_data.proto\x12\tmediapipe\x1a\x1amediapipe/util/color.proto\"\xbb\x01\n\nRenderData\x12J\n\x12render_annotations\x18\x01 \x03(\x0b\x32\x1b.mediapipe.RenderAnnotationR\x11renderAnnotations\x12\x1f\n\x0bscene_class\x18\x02 \x01(\tR\nsceneClass\x12@\n\x0escene_viewport\x18\x03 \x01(\x0b\x32\x19.mediapipe.RenderViewportR\rsceneViewport\"\xff\x17\n\x10RenderAnnotation\x12\x45\n\trectangle\x18\x01 \x01(\x0b\x32%.mediapipe.RenderAnnotation.RectangleH\x00R\trectangle\x12X\n\x10\x66illed_rectangle\x18\x02 \x01(\x0b\x32+.mediapipe.RenderAnnotation.FilledRectangleH\x00R\x0f\x66illedRectangle\x12\x36\n\x04oval\x18\x03 \x01(\x0b\x32 .mediapipe.RenderAnnotation.OvalH\x00R\x04oval\x12I\n\x0b\x66illed_oval\x18\x04 \x01(\x0b\x32&.mediapipe.RenderAnnotation.FilledOvalH\x00R\nfilledOval\x12\x39\n\x05point\x18\x05 \x01(\x0b\x32!.mediapipe.RenderAnnotation.PointH\x00R\x05point\x12\x36\n\x04line\x18\x06 \x01(\x0b\x32 .mediapipe.RenderAnnotation.LineH\x00R\x04line\x12\x39\n\x05\x61rrow\x18\x07 \x01(\x0b\x32!.mediapipe.RenderAnnotation.ArrowH\x00R\x05\x61rrow\x12\x36\n\x04text\x18\x08 \x01(\x0b\x32 .mediapipe.RenderAnnotation.TextH\x00R\x04text\x12[\n\x11rounded_rectangle\x18\t \x01(\x0b\x32,.mediapipe.RenderAnnotation.RoundedRectangleH\x00R\x10roundedRectangle\x12n\n\x18\x66illed_rounded_rectangle\x18\n \x01(\x0b\x32\x32.mediapipe.RenderAnnotation.FilledRoundedRectangleH\x00R\x16\x66illedRoundedRectangle\x12O\n\rgradient_line\x18\x0e \x01(\x0b\x32(.mediapipe.RenderAnnotation.GradientLineH\x00R\x0cgradientLine\x12\x42\n\x08scribble\x18\x0f \x01(\x0b\x32$.mediapipe.RenderAnnotation.ScribbleH\x00R\x08scribble\x12\x1f\n\tthickness\x18\x0b \x01(\x01:\x01\x31R\tthickness\x12&\n\x05\x63olor\x18\x0c \x01(\x0b\x32\x10.mediapipe.ColorR\x05\x63olor\x12\x1b\n\tscene_tag\x18\r \x01(\tR\x08sceneTag\x1a\xd0\x01\n\tRectangle\x12\x12\n\x04left\x18\x01 \x01(\x01R\x04left\x12\x10\n\x03top\x18\x02 \x01(\x01R\x03top\x12\x14\n\x05right\x18\x03 \x01(\x01R\x05right\x12\x16\n\x06\x62ottom\x18\x04 \x01(\x01R\x06\x62ottom\x12%\n\nnormalized\x18\x05 \x01(\x08:\x05\x66\x61lseR\nnormalized\x12\x1a\n\x08rotation\x18\x06 \x01(\x01R\x08rotation\x12,\n\x12top_left_thickness\x18\x07 \x01(\x01R\x10topLeftThickness\x1a\x87\x01\n\x0f\x46illedRectangle\x12\x43\n\trectangle\x18\x01 \x01(\x0b\x32%.mediapipe.RenderAnnotation.RectangleR\trectangle\x12/\n\nfill_color\x18\x02 \x01(\x0b\x32\x10.mediapipe.ColorR\tfillColor\x1a\x9f\x01\n\x10RoundedRectangle\x12\x43\n\trectangle\x18\x01 \x01(\x0b\x32%.mediapipe.RenderAnnotation.RectangleR\trectangle\x12&\n\rcorner_radius\x18\x02 \x01(\x05:\x01\x30R\x0c\x63ornerRadius\x12\x1e\n\tline_type\x18\x03 \x01(\x05:\x01\x34R\x08lineType\x1a\xa4\x01\n\x16\x46illedRoundedRectangle\x12Y\n\x11rounded_rectangle\x18\x01 \x01(\x0b\x32,.mediapipe.RenderAnnotation.RoundedRectangleR\x10roundedRectangle\x12/\n\nfill_color\x18\x02 \x01(\x0b\x32\x10.mediapipe.ColorR\tfillColor\x1aK\n\x04Oval\x12\x43\n\trectangle\x18\x01 \x01(\x0b\x32%.mediapipe.RenderAnnotation.RectangleR\trectangle\x1as\n\nFilledOval\x12\x34\n\x04oval\x18\x01 \x01(\x0b\x32 .mediapipe.RenderAnnotation.OvalR\x04oval\x12/\n\nfill_color\x18\x02 \x01(\x0b\x32\x10.mediapipe.ColorR\tfillColor\x1aJ\n\x05Point\x12\x0c\n\x01x\x18\x01 \x01(\x01R\x01x\x12\x0c\n\x01y\x18\x02 \x01(\x01R\x01y\x12%\n\nnormalized\x18\x03 \x01(\x08:\x05\x66\x61lseR\nnormalized\x1a\x88\x02\n\x04Line\x12\x17\n\x07x_start\x18\x01 \x01(\x01R\x06xStart\x12\x17\n\x07y_start\x18\x02 \x01(\x01R\x06yStart\x12\x13\n\x05x_end\x18\x03 \x01(\x01R\x04xEnd\x12\x13\n\x05y_end\x18\x04 \x01(\x01R\x04yEnd\x12%\n\nnormalized\x18\x05 \x01(\x08:\x05\x66\x61lseR\nnormalized\x12M\n\tline_type\x18\x06 \x01(\x0e\x32).mediapipe.RenderAnnotation.Line.LineType:\x05SOLIDR\x08lineType\".\n\x08LineType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\t\n\x05SOLID\x10\x01\x12\n\n\x06\x44\x41SHED\x10\x02\x1a\xe5\x01\n\x0cGradientLine\x12\x17\n\x07x_start\x18\x01 \x01(\x01R\x06xStart\x12\x17\n\x07y_start\x18\x02 \x01(\x01R\x06yStart\x12\x13\n\x05x_end\x18\x03 \x01(\x01R\x04xEnd\x12\x13\n\x05y_end\x18\x04 \x01(\x01R\x04yEnd\x12%\n\nnormalized\x18\x05 \x01(\x08:\x05\x66\x61lseR\nnormalized\x12(\n\x06\x63olor1\x18\x06 \x01(\x0b\x32\x10.mediapipe.ColorR\x06\x63olor1\x12(\n\x06\x63olor2\x18\x07 \x01(\x0b\x32\x10.mediapipe.ColorR\x06\x63olor2\x1a\x43\n\x08Scribble\x12\x37\n\x05point\x18\x01 \x03(\x0b\x32!.mediapipe.RenderAnnotation.PointR\x05point\x1a\x8a\x01\n\x05\x41rrow\x12\x17\n\x07x_start\x18\x01 \x01(\x01R\x06xStart\x12\x17\n\x07y_start\x18\x02 \x01(\x01R\x06yStart\x12\x13\n\x05x_end\x18\x03 \x01(\x01R\x04xEnd\x12\x13\n\x05y_end\x18\x04 \x01(\x01R\x04yEnd\x12%\n\nnormalized\x18\x05 \x01(\x08:\x05\x66\x61lseR\nnormalized\x1a\x97\x03\n\x04Text\x12!\n\x0c\x64isplay_text\x18\x01 \x01(\tR\x0b\x64isplayText\x12\x12\n\x04left\x18\x02 \x01(\x01R\x04left\x12\x1a\n\x08\x62\x61seline\x18\x03 \x01(\x01R\x08\x62\x61seline\x12\"\n\x0b\x66ont_height\x18\x04 \x01(\x01:\x01\x38R\nfontHeight\x12%\n\nnormalized\x18\x05 \x01(\x08:\x05\x66\x61lseR\nnormalized\x12\x1e\n\tfont_face\x18\x06 \x01(\x05:\x01\x30R\x08\x66ontFace\x12\x36\n\x13\x63\x65nter_horizontally\x18\x07 \x01(\x08:\x05\x66\x61lseR\x12\x63\x65nterHorizontally\x12\x32\n\x11\x63\x65nter_vertically\x18\x08 \x01(\x08:\x05\x66\x61lseR\x10\x63\x65nterVertically\x12.\n\x11outline_thickness\x18\x0b \x01(\x01:\x01\x30R\x10outlineThickness\x12\x35\n\routline_color\x18\x0c \x01(\x0b\x32\x10.mediapipe.ColorR\x0coutlineColorB\x06\n\x04\x64\x61ta\"\x82\x01\n\x0eRenderViewport\x12\x0e\n\x02id\x18\x01 \x01(\tR\x02id\x12\x19\n\x08width_px\x18\x02 \x01(\x05R\x07widthPx\x12\x1b\n\theight_px\x18\x03 \x01(\x05R\x08heightPx\x12(\n\x10\x63ompose_on_video\x18\x04 \x01(\x08R\x0e\x63omposeOnVideoB^\n\x1f\x63om.google.mediapipe.util.protoB\x0fRenderDataProtoZ*github.com/google/mediapipe/mediapipe/util')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.render_data_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\037com.google.mediapipe.util.protoB\017RenderDataProtoZ*github.com/google/mediapipe/mediapipe/util'
  _RENDERDATA._serialized_start=76
  _RENDERDATA._serialized_end=263
  _RENDERANNOTATION._serialized_start=266
  _RENDERANNOTATION._serialized_end=3337
  _RENDERANNOTATION_RECTANGLE._serialized_start=1265
  _RENDERANNOTATION_RECTANGLE._serialized_end=1473
  _RENDERANNOTATION_FILLEDRECTANGLE._serialized_start=1476
  _RENDERANNOTATION_FILLEDRECTANGLE._serialized_end=1611
  _RENDERANNOTATION_ROUNDEDRECTANGLE._serialized_start=1614
  _RENDERANNOTATION_ROUNDEDRECTANGLE._serialized_end=1773
  _RENDERANNOTATION_FILLEDROUNDEDRECTANGLE._serialized_start=1776
  _RENDERANNOTATION_FILLEDROUNDEDRECTANGLE._serialized_end=1940
  _RENDERANNOTATION_OVAL._serialized_start=1942
  _RENDERANNOTATION_OVAL._serialized_end=2017
  _RENDERANNOTATION_FILLEDOVAL._serialized_start=2019
  _RENDERANNOTATION_FILLEDOVAL._serialized_end=2134
  _RENDERANNOTATION_POINT._serialized_start=2136
  _RENDERANNOTATION_POINT._serialized_end=2210
  _RENDERANNOTATION_LINE._serialized_start=2213
  _RENDERANNOTATION_LINE._serialized_end=2477
  _RENDERANNOTATION_LINE_LINETYPE._serialized_start=2431
  _RENDERANNOTATION_LINE_LINETYPE._serialized_end=2477
  _RENDERANNOTATION_GRADIENTLINE._serialized_start=2480
  _RENDERANNOTATION_GRADIENTLINE._serialized_end=2709
  _RENDERANNOTATION_SCRIBBLE._serialized_start=2711
  _RENDERANNOTATION_SCRIBBLE._serialized_end=2778
  _RENDERANNOTATION_ARROW._serialized_start=2781
  _RENDERANNOTATION_ARROW._serialized_end=2919
  _RENDERANNOTATION_TEXT._serialized_start=2922
  _RENDERANNOTATION_TEXT._serialized_end=3329
  _RENDERVIEWPORT._serialized_start=3340
  _RENDERVIEWPORT._serialized_end=3470
# @@protoc_insertion_point(module_scope)
