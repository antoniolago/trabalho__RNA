import axios, { AxiosError } from 'axios';
import { CONFIG } from '../configs';
import Cookies from "js-cookie";
import jwt_decode from "jwt-decode";

interface IToken {
  usuarioProfessor: string;
  isImpersonating: string;
  role: string[];
  membroDe: string[];
  nome: string;
  username: string;
}

export const AuthCookieName = "auth_token";
export const useApi = () => {
  const getApiUrl = () => {
    let url;
    var pathname = window.location.href
    // console.log(CONFIG.GATEWAY_URL)
    if(pathname.includes("localhost")){
      url = 'http://rna-algarismos.lag0.com.br:5001';
    } else if(pathname.includes("-stg")){
      url = "https://rna-algarismos-stg.lag0.com.br";
    } else {
      url = "https://rna-algarismos.lag0.com.br";
    }
    return url;
  }
  const api = axios.create({
    baseURL: getApiUrl(),
  });

  api.interceptors.request.use(async (config) => {
    // const token = Cookies.get(AuthCookieName);

    // if (token) {
    //   config.headers["Authorization"] = `Bearer ${token}`;
    // }

    return config;
  });

  api.interceptors.response.use(
    function (response) {
      return response;
    },
    function (error: AxiosError) {
      if (401 === error?.response?.status) {
        logout();
        // alert("Houve um problema de autenticação, por favor logue novamente.");
        // if(window.location.href.includes("/login"))
        //   window.location.href = "/login";
      } else {
        return Promise.reject(error);
      }
    }
  );
  const setToken = (token: string) => {
    Cookies.set(AuthCookieName, token, { domain: "rna-algarismos.lag0.com.br" });
  };
  const getToken = () => {
    return Cookies.get(AuthCookieName);
  };

  const getDecodedToken = (): IToken | undefined => {
    const { getToken } = useApi();
    const token = getToken();
    if(token){
      return jwt_decode(token!);
    }
    return undefined;
  }
  
  const logout = () => {
    Cookies.remove(AuthCookieName, { domain: "rna-algarismos.lag0.com.br" });
    // window.location.href = "/login";
  }

  const isLogado = () => {
    const token = getDecodedToken();
    return token?.usuarioProfessor != undefined;
  }

  const temRole = (role: string): boolean => {
    const token = Cookies.get(AuthCookieName);
    if (token != null) {
      var decodedToken = jwt_decode(token as string) as { role: string[] };
      if (decodedToken.role.includes(role)) 
        return true;
      else 
        return false;
    }
    return false;
  }

  return { api, isLogado, getToken, setToken, logout, getDecodedToken, temRole };
};