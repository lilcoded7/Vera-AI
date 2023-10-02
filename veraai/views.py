from rest_framework.generics import CreateAPIView
from rest_framework import status, generics
from rest_framework.permissions import AllowAny
from .models import *
from .serializers import *
from rest_framework.response import Response


class VINSINCBUILDINGDATACreateUpdateView(CreateAPIView):
    serializer_class = VINSINCBUILDINGDATASerializer
    queryset = VINSINCBUILDINGDATA.objects.all()
    def perform_create(self, serializer):
        unique_identifier = serializer.validated_data.get('timestamp')
        existing_instance = VINSINCBUILDINGDATA.objects.filter(timestamp=unique_identifier).first()
        if existing_instance:
            serializer.update(existing_instance, serializer.validated_data)
        else:
            serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class BuildingAPIVIew(generics.GenericAPIView):
    serializer_class = BuildingSerializer
    permission_classes = [AllowAny]
    def post(self, request):
        data = self.serializer_class(data=request.data)
        if data.is_valid():
            data.save()
            return Response({'message':'Building added!'})
        return Response({'message':'builing faild to add'})


class UploadBuildingDataCreateUpdateView(CreateAPIView):
    serializer_class = UploadBuildingDataSerializer
    queryset = UploadBuildingData.objects.all()
    def perform_create(self, serializer):
        building = serializer.validated_data.get('building')
        existing_instance = UploadBuildingData.objects.filter(building=building).first()
        if existing_instance:
            serializer.update(existing_instance, serializer.validated_data)
        else:
            serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    



