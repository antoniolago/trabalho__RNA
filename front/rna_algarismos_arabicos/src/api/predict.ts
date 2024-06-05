import { useMutation, UseMutationOptions, useQuery, UseQueryOptions } from "@tanstack/react-query";
import { useApi } from ".";
import { AxiosError, AxiosResponse } from "axios";
import { toast } from "sonner";

const useGetPrediction = () => {
    const { api } = useApi();

    var queryOptions: UseQueryOptions<any,
        Error,
        any,
        any> = {
        queryKey: ["Predicts"],
        queryFn: (image: any) => {
            const formData = new FormData();
            formData.append('image', image);
            api.post("predict", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            })
        },
        staleTime: 600000,
    };
    const context = useQuery(queryOptions)
    return { ...context, data: context.data?.data };
};

const useMutatePredict = (teste: any) => {
    const { api, getDecodedToken } = useApi()

    var mutationOptions: UseMutationOptions = {
        mutationFn: () =>
            api.post("predict",
                {
                    image: teste
                })
                .then((_response: any) => {
                    //   toast.success('Predict.')
                })
                .catch((response: AxiosError) => {
                    toast.error('Predict error.')
                })
    };
    return useMutation(mutationOptions)
}
export const PredictionService = {
    useGetPrediction,
};